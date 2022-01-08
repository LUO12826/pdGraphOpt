//===- AArch64MIPeepholeOpt.cpp - AArch64 MI peephole optimization pass ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs below peephole optimizations on MIR level.
//
// 1. MOVi32imm + ANDWrr ==> ANDWri + ANDWri
//    MOVi64imm + ANDXrr ==> ANDXri + ANDXri
//
//    The mov pseudo instruction could be expanded to multiple mov instructions
//    later. In this case, we could try to split the constant  operand of mov
//    instruction into two bitmask immediates. It makes two AND instructions
//    intead of multiple `mov` + `and` instructions.
//
// 2. Remove redundant ORRWrs which is generated by zero-extend.
//
//    %3:gpr32 = ORRWrs $wzr, %2, 0
//    %4:gpr64 = SUBREG_TO_REG 0, %3, %subreg.sub_32
//
//    If AArch64's 32-bit form of instruction defines the source operand of
//    ORRWrs, we can remove the ORRWrs because the upper 32 bits of the source
//    operand are set to zero.
//
//===----------------------------------------------------------------------===//

#include "AArch64ExpandImm.h"
#include "AArch64InstrInfo.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLoopInfo.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-mi-peephole-opt"

namespace {

struct AArch64MIPeepholeOpt : public MachineFunctionPass {
  static char ID;

  AArch64MIPeepholeOpt() : MachineFunctionPass(ID) {
    initializeAArch64MIPeepholeOptPass(*PassRegistry::getPassRegistry());
  }

  const AArch64InstrInfo *TII;
  MachineLoopInfo *MLI;
  MachineRegisterInfo *MRI;

  template <typename T>
  bool visitAND(MachineInstr &MI,
                SmallSetVector<MachineInstr *, 8> &ToBeRemoved);
  bool visitORR(MachineInstr &MI,
                SmallSetVector<MachineInstr *, 8> &ToBeRemoved);
  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AArch64 MI Peephole Optimization pass";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineLoopInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

char AArch64MIPeepholeOpt::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS(AArch64MIPeepholeOpt, "aarch64-mi-peephole-opt",
                "AArch64 MI Peephole Optimization", false, false)

template <typename T>
static bool splitBitmaskImm(T Imm, unsigned RegSize, T &Imm1Enc, T &Imm2Enc) {
  T UImm = static_cast<T>(Imm);
  if (AArch64_AM::isLogicalImmediate(UImm, RegSize))
    return false;

  // If this immediate can be handled by one instruction, do not split it.
  SmallVector<AArch64_IMM::ImmInsnModel, 4> Insn;
  AArch64_IMM::expandMOVImm(UImm, RegSize, Insn);
  if (Insn.size() == 1)
    return false;

  // The bitmask immediate consists of consecutive ones.  Let's say there is
  // constant 0b00000000001000000000010000000000 which does not consist of
  // consecutive ones. We can split it in to two bitmask immediate like
  // 0b00000000001111111111110000000000 and 0b11111111111000000000011111111111.
  // If we do AND with these two bitmask immediate, we can see original one.
  unsigned LowestBitSet = countTrailingZeros(UImm);
  unsigned HighestBitSet = Log2_64(UImm);

  // Create a mask which is filled with one from the position of lowest bit set
  // to the position of highest bit set.
  T NewImm1 = (static_cast<T>(2) << HighestBitSet) -
              (static_cast<T>(1) << LowestBitSet);
  // Create a mask which is filled with one outside the position of lowest bit
  // set and the position of highest bit set.
  T NewImm2 = UImm | ~NewImm1;

  // If the split value is not valid bitmask immediate, do not split this
  // constant.
  if (!AArch64_AM::isLogicalImmediate(NewImm2, RegSize))
    return false;

  Imm1Enc = AArch64_AM::encodeLogicalImmediate(NewImm1, RegSize);
  Imm2Enc = AArch64_AM::encodeLogicalImmediate(NewImm2, RegSize);
  return true;
}

template <typename T>
bool AArch64MIPeepholeOpt::visitAND(
    MachineInstr &MI, SmallSetVector<MachineInstr *, 8> &ToBeRemoved) {
  // Try below transformation.
  //
  // MOVi32imm + ANDWrr ==> ANDWri + ANDWri
  // MOVi64imm + ANDXrr ==> ANDXri + ANDXri
  //
  // The mov pseudo instruction could be expanded to multiple mov instructions
  // later. Let's try to split the constant operand of mov instruction into two
  // bitmask immediates. It makes only two AND instructions intead of multiple
  // mov + and instructions.

  unsigned RegSize = sizeof(T) * 8;
  assert((RegSize == 32 || RegSize == 64) &&
         "Invalid RegSize for AND bitmask peephole optimization");

  // Check whether AND's MBB is in loop and the AND is loop invariant.
  MachineBasicBlock *MBB = MI.getParent();
  MachineLoop *L = MLI->getLoopFor(MBB);
  if (L && !L->isLoopInvariant(MI))
    return false;

  // Check whether AND's operand is MOV with immediate.
  MachineInstr *MovMI = MRI->getUniqueVRegDef(MI.getOperand(2).getReg());
  if (!MovMI)
    return false;

  MachineInstr *SubregToRegMI = nullptr;
  // If it is SUBREG_TO_REG, check its operand.
  if (MovMI->getOpcode() == TargetOpcode::SUBREG_TO_REG) {
    SubregToRegMI = MovMI;
    MovMI = MRI->getUniqueVRegDef(MovMI->getOperand(2).getReg());
    if (!MovMI)
      return false;
  }

  if (MovMI->getOpcode() != AArch64::MOVi32imm &&
      MovMI->getOpcode() != AArch64::MOVi64imm)
    return false;

  // If the MOV has multiple uses, do not split the immediate because it causes
  // more instructions.
  if (!MRI->hasOneUse(MovMI->getOperand(0).getReg()))
    return false;

  if (SubregToRegMI && !MRI->hasOneUse(SubregToRegMI->getOperand(0).getReg()))
    return false;

  // Split the bitmask immediate into two.
  T UImm = static_cast<T>(MovMI->getOperand(1).getImm());
  // For the 32 bit form of instruction, the upper 32 bits of the destination
  // register are set to zero. If there is SUBREG_TO_REG, set the upper 32 bits
  // of UImm to zero.
  if (SubregToRegMI)
    UImm &= 0xFFFFFFFF;
  T Imm1Enc;
  T Imm2Enc;
  if (!splitBitmaskImm(UImm, RegSize, Imm1Enc, Imm2Enc))
    return false;

  // Create new AND MIs.
  DebugLoc DL = MI.getDebugLoc();
  const TargetRegisterClass *ANDImmRC =
      (RegSize == 32) ? &AArch64::GPR32spRegClass : &AArch64::GPR64spRegClass;
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  Register NewTmpReg = MRI->createVirtualRegister(ANDImmRC);
  Register NewDstReg = MRI->createVirtualRegister(ANDImmRC);
  unsigned Opcode = (RegSize == 32) ? AArch64::ANDWri : AArch64::ANDXri;

  MRI->constrainRegClass(NewTmpReg, MRI->getRegClass(SrcReg));
  BuildMI(*MBB, MI, DL, TII->get(Opcode), NewTmpReg)
      .addReg(SrcReg)
      .addImm(Imm1Enc);

  MRI->constrainRegClass(NewDstReg, MRI->getRegClass(DstReg));
  BuildMI(*MBB, MI, DL, TII->get(Opcode), NewDstReg)
      .addReg(NewTmpReg)
      .addImm(Imm2Enc);

  MRI->replaceRegWith(DstReg, NewDstReg);
  // replaceRegWith changes MI's definition register. Keep it for SSA form until
  // deleting MI.
  MI.getOperand(0).setReg(DstReg);

  ToBeRemoved.insert(&MI);
  if (SubregToRegMI)
    ToBeRemoved.insert(SubregToRegMI);
  ToBeRemoved.insert(MovMI);

  return true;
}

bool AArch64MIPeepholeOpt::visitORR(
    MachineInstr &MI, SmallSetVector<MachineInstr *, 8> &ToBeRemoved) {
  // Check this ORR comes from below zero-extend pattern.
  //
  // def : Pat<(i64 (zext GPR32:$src)),
  //           (SUBREG_TO_REG (i32 0), (ORRWrs WZR, GPR32:$src, 0), sub_32)>;
  if (MI.getOperand(3).getImm() != 0)
    return false;

  if (MI.getOperand(1).getReg() != AArch64::WZR)
    return false;

  MachineInstr *SrcMI = MRI->getUniqueVRegDef(MI.getOperand(2).getReg());
  if (!SrcMI)
    return false;

  // From https://developer.arm.com/documentation/dui0801/b/BABBGCAC
  //
  // When you use the 32-bit form of an instruction, the upper 32 bits of the
  // source registers are ignored and the upper 32 bits of the destination
  // register are set to zero.
  //
  // If AArch64's 32-bit form of instruction defines the source operand of
  // zero-extend, we do not need the zero-extend. Let's check the MI's opcode is
  // real AArch64 instruction and if it is not, do not process the opcode
  // conservatively.
  if (SrcMI->getOpcode() <= TargetOpcode::GENERIC_OP_END)
    return false;

  Register DefReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(2).getReg();
  MRI->replaceRegWith(DefReg, SrcReg);
  MRI->clearKillFlags(SrcReg);
  // replaceRegWith changes MI's definition register. Keep it for SSA form until
  // deleting MI.
  MI.getOperand(0).setReg(DefReg);
  ToBeRemoved.insert(&MI);

  LLVM_DEBUG(dbgs() << "Removed: " << MI << "\n");

  return true;
}

bool AArch64MIPeepholeOpt::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  TII = static_cast<const AArch64InstrInfo *>(MF.getSubtarget().getInstrInfo());
  MLI = &getAnalysis<MachineLoopInfo>();
  MRI = &MF.getRegInfo();

  assert (MRI->isSSA() && "Expected to be run on SSA form!");

  bool Changed = false;
  SmallSetVector<MachineInstr *, 8> ToBeRemoved;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      switch (MI.getOpcode()) {
      default:
        break;
      case AArch64::ANDWrr:
        Changed = visitAND<uint32_t>(MI, ToBeRemoved);
        break;
      case AArch64::ANDXrr:
        Changed = visitAND<uint64_t>(MI, ToBeRemoved);
        break;
      case AArch64::ORRWrs:
        Changed = visitORR(MI, ToBeRemoved);
        break;
      }
    }
  }

  for (MachineInstr *MI : ToBeRemoved)
    MI->eraseFromParent();

  return Changed;
}

FunctionPass *llvm::createAArch64MIPeepholeOptPass() {
  return new AArch64MIPeepholeOpt();
}
