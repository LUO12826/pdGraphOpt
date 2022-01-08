; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv32 -mattr=+d -verify-machineinstrs < %s \
; RUN:   -disable-strictnode-mutation | FileCheck -check-prefix=RV32IFD %s
; RUN: llc -mtriple=riscv64 -mattr=+d -verify-machineinstrs < %s \
; RUN:   -disable-strictnode-mutation | FileCheck -check-prefix=RV64IFD %s
; RUN: llc -mtriple=riscv32 -verify-machineinstrs < %s \
; RUN:   -disable-strictnode-mutation | FileCheck -check-prefix=RV32I %s
; RUN: llc -mtriple=riscv64 -verify-machineinstrs < %s \
; RUN:   -disable-strictnode-mutation | FileCheck -check-prefix=RV64I %s

; NOTE: The rounding mode metadata does not effect which instruction is
; selected. Dynamic rounding mode is always used for operations that
; support rounding mode.

define float @fcvt_s_d(double %a) nounwind strictfp {
; RV32IFD-LABEL: fcvt_s_d:
; RV32IFD:       # %bb.0:
; RV32IFD-NEXT:    addi sp, sp, -16
; RV32IFD-NEXT:    sw a0, 8(sp)
; RV32IFD-NEXT:    sw a1, 12(sp)
; RV32IFD-NEXT:    fld ft0, 8(sp)
; RV32IFD-NEXT:    fcvt.s.d ft0, ft0
; RV32IFD-NEXT:    fmv.x.w a0, ft0
; RV32IFD-NEXT:    addi sp, sp, 16
; RV32IFD-NEXT:    ret
;
; RV64IFD-LABEL: fcvt_s_d:
; RV64IFD:       # %bb.0:
; RV64IFD-NEXT:    fmv.d.x ft0, a0
; RV64IFD-NEXT:    fcvt.s.d ft0, ft0
; RV64IFD-NEXT:    fmv.x.w a0, ft0
; RV64IFD-NEXT:    ret
;
; RV32I-LABEL: fcvt_s_d:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -16
; RV32I-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32I-NEXT:    call __truncdfsf2@plt
; RV32I-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32I-NEXT:    addi sp, sp, 16
; RV32I-NEXT:    ret
;
; RV64I-LABEL: fcvt_s_d:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -16
; RV64I-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64I-NEXT:    call __truncdfsf2@plt
; RV64I-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64I-NEXT:    addi sp, sp, 16
; RV64I-NEXT:    ret
  %1 = call float @llvm.experimental.constrained.fptrunc.f32.f64(double %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %1
}
declare float @llvm.experimental.constrained.fptrunc.f32.f64(double, metadata, metadata)

define double @fcvt_d_s(float %a) nounwind strictfp {
; RV32IFD-LABEL: fcvt_d_s:
; RV32IFD:       # %bb.0:
; RV32IFD-NEXT:    addi sp, sp, -16
; RV32IFD-NEXT:    fmv.w.x ft0, a0
; RV32IFD-NEXT:    fcvt.d.s ft0, ft0
; RV32IFD-NEXT:    fsd ft0, 8(sp)
; RV32IFD-NEXT:    lw a0, 8(sp)
; RV32IFD-NEXT:    lw a1, 12(sp)
; RV32IFD-NEXT:    addi sp, sp, 16
; RV32IFD-NEXT:    ret
;
; RV64IFD-LABEL: fcvt_d_s:
; RV64IFD:       # %bb.0:
; RV64IFD-NEXT:    fmv.w.x ft0, a0
; RV64IFD-NEXT:    fcvt.d.s ft0, ft0
; RV64IFD-NEXT:    fmv.x.d a0, ft0
; RV64IFD-NEXT:    ret
;
; RV32I-LABEL: fcvt_d_s:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -16
; RV32I-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32I-NEXT:    call __extendsfdf2@plt
; RV32I-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32I-NEXT:    addi sp, sp, 16
; RV32I-NEXT:    ret
;
; RV64I-LABEL: fcvt_d_s:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -16
; RV64I-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64I-NEXT:    call __extendsfdf2@plt
; RV64I-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64I-NEXT:    addi sp, sp, 16
; RV64I-NEXT:    ret
  %1 = call double @llvm.experimental.constrained.fpext.f64.f32(float %a, metadata !"fpexcept.strict")
  ret double %1
}
declare double @llvm.experimental.constrained.fpext.f64.f32(float, metadata)

define i32 @fcvt_w_d(double %a) nounwind strictfp {
; RV32IFD-LABEL: fcvt_w_d:
; RV32IFD:       # %bb.0:
; RV32IFD-NEXT:    addi sp, sp, -16
; RV32IFD-NEXT:    sw a0, 8(sp)
; RV32IFD-NEXT:    sw a1, 12(sp)
; RV32IFD-NEXT:    fld ft0, 8(sp)
; RV32IFD-NEXT:    fcvt.w.d a0, ft0, rtz
; RV32IFD-NEXT:    addi sp, sp, 16
; RV32IFD-NEXT:    ret
;
; RV64IFD-LABEL: fcvt_w_d:
; RV64IFD:       # %bb.0:
; RV64IFD-NEXT:    fmv.d.x ft0, a0
; RV64IFD-NEXT:    fcvt.w.d a0, ft0, rtz
; RV64IFD-NEXT:    ret
;
; RV32I-LABEL: fcvt_w_d:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -16
; RV32I-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32I-NEXT:    call __fixdfsi@plt
; RV32I-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32I-NEXT:    addi sp, sp, 16
; RV32I-NEXT:    ret
;
; RV64I-LABEL: fcvt_w_d:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -16
; RV64I-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64I-NEXT:    call __fixdfsi@plt
; RV64I-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64I-NEXT:    addi sp, sp, 16
; RV64I-NEXT:    ret
  %1 = call i32 @llvm.experimental.constrained.fptosi.i32.f64(double %a, metadata !"fpexcept.strict") strictfp
  ret i32 %1
}
declare i32 @llvm.experimental.constrained.fptosi.i32.f64(double, metadata)

; For RV64D, fcvt.lu.d is semantically equivalent to fcvt.wu.d in this case
; because fptosi will produce poison if the result doesn't fit into an i32.
define i32 @fcvt_wu_d(double %a) nounwind strictfp {
; RV32IFD-LABEL: fcvt_wu_d:
; RV32IFD:       # %bb.0:
; RV32IFD-NEXT:    addi sp, sp, -16
; RV32IFD-NEXT:    sw a0, 8(sp)
; RV32IFD-NEXT:    sw a1, 12(sp)
; RV32IFD-NEXT:    fld ft0, 8(sp)
; RV32IFD-NEXT:    fcvt.wu.d a0, ft0, rtz
; RV32IFD-NEXT:    addi sp, sp, 16
; RV32IFD-NEXT:    ret
;
; RV64IFD-LABEL: fcvt_wu_d:
; RV64IFD:       # %bb.0:
; RV64IFD-NEXT:    fmv.d.x ft0, a0
; RV64IFD-NEXT:    fcvt.wu.d a0, ft0, rtz
; RV64IFD-NEXT:    ret
;
; RV32I-LABEL: fcvt_wu_d:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -16
; RV32I-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32I-NEXT:    call __fixunsdfsi@plt
; RV32I-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32I-NEXT:    addi sp, sp, 16
; RV32I-NEXT:    ret
;
; RV64I-LABEL: fcvt_wu_d:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -16
; RV64I-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64I-NEXT:    call __fixunsdfsi@plt
; RV64I-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64I-NEXT:    addi sp, sp, 16
; RV64I-NEXT:    ret
  %1 = call i32 @llvm.experimental.constrained.fptoui.i32.f64(double %a, metadata !"fpexcept.strict") strictfp
  ret i32 %1
}
declare i32 @llvm.experimental.constrained.fptoui.i32.f64(double, metadata)

; Test where the fptoui has multiple uses, one of which causes a sext to be
; inserted on RV64.
define i32 @fcvt_wu_d_multiple_use(double %x, i32* %y) {
; RV32IFD-LABEL: fcvt_wu_d_multiple_use:
; RV32IFD:       # %bb.0:
; RV32IFD-NEXT:    addi sp, sp, -16
; RV32IFD-NEXT:    .cfi_def_cfa_offset 16
; RV32IFD-NEXT:    sw a0, 8(sp)
; RV32IFD-NEXT:    sw a1, 12(sp)
; RV32IFD-NEXT:    fld ft0, 8(sp)
; RV32IFD-NEXT:    fcvt.wu.d a1, ft0, rtz
; RV32IFD-NEXT:    li a0, 1
; RV32IFD-NEXT:    beqz a1, .LBB4_2
; RV32IFD-NEXT:  # %bb.1:
; RV32IFD-NEXT:    mv a0, a1
; RV32IFD-NEXT:  .LBB4_2:
; RV32IFD-NEXT:    addi sp, sp, 16
; RV32IFD-NEXT:    ret
;
; RV64IFD-LABEL: fcvt_wu_d_multiple_use:
; RV64IFD:       # %bb.0:
; RV64IFD-NEXT:    fmv.d.x ft0, a0
; RV64IFD-NEXT:    fcvt.wu.d a1, ft0, rtz
; RV64IFD-NEXT:    li a0, 1
; RV64IFD-NEXT:    beqz a1, .LBB4_2
; RV64IFD-NEXT:  # %bb.1:
; RV64IFD-NEXT:    mv a0, a1
; RV64IFD-NEXT:  .LBB4_2:
; RV64IFD-NEXT:    ret
;
; RV32I-LABEL: fcvt_wu_d_multiple_use:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -16
; RV32I-NEXT:    .cfi_def_cfa_offset 16
; RV32I-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32I-NEXT:    .cfi_offset ra, -4
; RV32I-NEXT:    call __fixunsdfsi@plt
; RV32I-NEXT:    mv a1, a0
; RV32I-NEXT:    li a0, 1
; RV32I-NEXT:    beqz a1, .LBB4_2
; RV32I-NEXT:  # %bb.1:
; RV32I-NEXT:    mv a0, a1
; RV32I-NEXT:  .LBB4_2:
; RV32I-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32I-NEXT:    addi sp, sp, 16
; RV32I-NEXT:    ret
;
; RV64I-LABEL: fcvt_wu_d_multiple_use:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -16
; RV64I-NEXT:    .cfi_def_cfa_offset 16
; RV64I-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64I-NEXT:    .cfi_offset ra, -8
; RV64I-NEXT:    call __fixunsdfsi@plt
; RV64I-NEXT:    mv a1, a0
; RV64I-NEXT:    li a0, 1
; RV64I-NEXT:    beqz a1, .LBB4_2
; RV64I-NEXT:  # %bb.1:
; RV64I-NEXT:    mv a0, a1
; RV64I-NEXT:  .LBB4_2:
; RV64I-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64I-NEXT:    addi sp, sp, 16
; RV64I-NEXT:    ret
  %a = call i32 @llvm.experimental.constrained.fptoui.i32.f64(double %x, metadata !"fpexcept.strict") strictfp
  %b = icmp eq i32 %a, 0
  %c = select i1 %b, i32 1, i32 %a
  ret i32 %c
}

define double @fcvt_d_w(i32 %a) nounwind strictfp {
; RV32IFD-LABEL: fcvt_d_w:
; RV32IFD:       # %bb.0:
; RV32IFD-NEXT:    addi sp, sp, -16
; RV32IFD-NEXT:    fcvt.d.w ft0, a0
; RV32IFD-NEXT:    fsd ft0, 8(sp)
; RV32IFD-NEXT:    lw a0, 8(sp)
; RV32IFD-NEXT:    lw a1, 12(sp)
; RV32IFD-NEXT:    addi sp, sp, 16
; RV32IFD-NEXT:    ret
;
; RV64IFD-LABEL: fcvt_d_w:
; RV64IFD:       # %bb.0:
; RV64IFD-NEXT:    fcvt.d.w ft0, a0
; RV64IFD-NEXT:    fmv.x.d a0, ft0
; RV64IFD-NEXT:    ret
;
; RV32I-LABEL: fcvt_d_w:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -16
; RV32I-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32I-NEXT:    call __floatsidf@plt
; RV32I-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32I-NEXT:    addi sp, sp, 16
; RV32I-NEXT:    ret
;
; RV64I-LABEL: fcvt_d_w:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -16
; RV64I-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64I-NEXT:    sext.w a0, a0
; RV64I-NEXT:    call __floatsidf@plt
; RV64I-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64I-NEXT:    addi sp, sp, 16
; RV64I-NEXT:    ret
  %1 = call double @llvm.experimental.constrained.sitofp.f64.i32(i32 %a, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret double %1
}
declare double @llvm.experimental.constrained.sitofp.f64.i32(i32, metadata, metadata)

define double @fcvt_d_w_load(i32* %p) nounwind strictfp {
; RV32IFD-LABEL: fcvt_d_w_load:
; RV32IFD:       # %bb.0:
; RV32IFD-NEXT:    addi sp, sp, -16
; RV32IFD-NEXT:    lw a0, 0(a0)
; RV32IFD-NEXT:    fcvt.d.w ft0, a0
; RV32IFD-NEXT:    fsd ft0, 8(sp)
; RV32IFD-NEXT:    lw a0, 8(sp)
; RV32IFD-NEXT:    lw a1, 12(sp)
; RV32IFD-NEXT:    addi sp, sp, 16
; RV32IFD-NEXT:    ret
;
; RV64IFD-LABEL: fcvt_d_w_load:
; RV64IFD:       # %bb.0:
; RV64IFD-NEXT:    lw a0, 0(a0)
; RV64IFD-NEXT:    fcvt.d.w ft0, a0
; RV64IFD-NEXT:    fmv.x.d a0, ft0
; RV64IFD-NEXT:    ret
;
; RV32I-LABEL: fcvt_d_w_load:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -16
; RV32I-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32I-NEXT:    lw a0, 0(a0)
; RV32I-NEXT:    call __floatsidf@plt
; RV32I-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32I-NEXT:    addi sp, sp, 16
; RV32I-NEXT:    ret
;
; RV64I-LABEL: fcvt_d_w_load:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -16
; RV64I-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64I-NEXT:    lw a0, 0(a0)
; RV64I-NEXT:    call __floatsidf@plt
; RV64I-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64I-NEXT:    addi sp, sp, 16
; RV64I-NEXT:    ret
  %a = load i32, i32* %p
  %1 = call double @llvm.experimental.constrained.sitofp.f64.i32(i32 %a, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret double %1
}

define double @fcvt_d_wu(i32 %a) nounwind strictfp {
; RV32IFD-LABEL: fcvt_d_wu:
; RV32IFD:       # %bb.0:
; RV32IFD-NEXT:    addi sp, sp, -16
; RV32IFD-NEXT:    fcvt.d.wu ft0, a0
; RV32IFD-NEXT:    fsd ft0, 8(sp)
; RV32IFD-NEXT:    lw a0, 8(sp)
; RV32IFD-NEXT:    lw a1, 12(sp)
; RV32IFD-NEXT:    addi sp, sp, 16
; RV32IFD-NEXT:    ret
;
; RV64IFD-LABEL: fcvt_d_wu:
; RV64IFD:       # %bb.0:
; RV64IFD-NEXT:    fcvt.d.wu ft0, a0
; RV64IFD-NEXT:    fmv.x.d a0, ft0
; RV64IFD-NEXT:    ret
;
; RV32I-LABEL: fcvt_d_wu:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -16
; RV32I-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32I-NEXT:    call __floatunsidf@plt
; RV32I-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32I-NEXT:    addi sp, sp, 16
; RV32I-NEXT:    ret
;
; RV64I-LABEL: fcvt_d_wu:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -16
; RV64I-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64I-NEXT:    sext.w a0, a0
; RV64I-NEXT:    call __floatunsidf@plt
; RV64I-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64I-NEXT:    addi sp, sp, 16
; RV64I-NEXT:    ret
  %1 = call double @llvm.experimental.constrained.uitofp.f64.i32(i32 %a, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret double %1
}
declare double @llvm.experimental.constrained.uitofp.f64.i32(i32, metadata, metadata)

define double @fcvt_d_wu_load(i32* %p) nounwind strictfp {
; RV32IFD-LABEL: fcvt_d_wu_load:
; RV32IFD:       # %bb.0:
; RV32IFD-NEXT:    addi sp, sp, -16
; RV32IFD-NEXT:    lw a0, 0(a0)
; RV32IFD-NEXT:    fcvt.d.wu ft0, a0
; RV32IFD-NEXT:    fsd ft0, 8(sp)
; RV32IFD-NEXT:    lw a0, 8(sp)
; RV32IFD-NEXT:    lw a1, 12(sp)
; RV32IFD-NEXT:    addi sp, sp, 16
; RV32IFD-NEXT:    ret
;
; RV64IFD-LABEL: fcvt_d_wu_load:
; RV64IFD:       # %bb.0:
; RV64IFD-NEXT:    lwu a0, 0(a0)
; RV64IFD-NEXT:    fcvt.d.wu ft0, a0
; RV64IFD-NEXT:    fmv.x.d a0, ft0
; RV64IFD-NEXT:    ret
;
; RV32I-LABEL: fcvt_d_wu_load:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -16
; RV32I-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32I-NEXT:    lw a0, 0(a0)
; RV32I-NEXT:    call __floatunsidf@plt
; RV32I-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32I-NEXT:    addi sp, sp, 16
; RV32I-NEXT:    ret
;
; RV64I-LABEL: fcvt_d_wu_load:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -16
; RV64I-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64I-NEXT:    lw a0, 0(a0)
; RV64I-NEXT:    call __floatunsidf@plt
; RV64I-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64I-NEXT:    addi sp, sp, 16
; RV64I-NEXT:    ret
  %a = load i32, i32* %p
  %1 = call double @llvm.experimental.constrained.uitofp.f64.i32(i32 %a, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret double %1
}

define i64 @fcvt_l_d(double %a) nounwind strictfp {
; RV32IFD-LABEL: fcvt_l_d:
; RV32IFD:       # %bb.0:
; RV32IFD-NEXT:    addi sp, sp, -16
; RV32IFD-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32IFD-NEXT:    call __fixdfdi@plt
; RV32IFD-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32IFD-NEXT:    addi sp, sp, 16
; RV32IFD-NEXT:    ret
;
; RV64IFD-LABEL: fcvt_l_d:
; RV64IFD:       # %bb.0:
; RV64IFD-NEXT:    fmv.d.x ft0, a0
; RV64IFD-NEXT:    fcvt.l.d a0, ft0, rtz
; RV64IFD-NEXT:    ret
;
; RV32I-LABEL: fcvt_l_d:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -16
; RV32I-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32I-NEXT:    call __fixdfdi@plt
; RV32I-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32I-NEXT:    addi sp, sp, 16
; RV32I-NEXT:    ret
;
; RV64I-LABEL: fcvt_l_d:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -16
; RV64I-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64I-NEXT:    call __fixdfdi@plt
; RV64I-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64I-NEXT:    addi sp, sp, 16
; RV64I-NEXT:    ret
  %1 = call i64 @llvm.experimental.constrained.fptosi.i64.f64(double %a, metadata !"fpexcept.strict") strictfp
  ret i64 %1
}
declare i64 @llvm.experimental.constrained.fptosi.i64.f64(double, metadata)

define i64 @fcvt_lu_d(double %a) nounwind strictfp {
; RV32IFD-LABEL: fcvt_lu_d:
; RV32IFD:       # %bb.0:
; RV32IFD-NEXT:    addi sp, sp, -16
; RV32IFD-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32IFD-NEXT:    call __fixunsdfdi@plt
; RV32IFD-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32IFD-NEXT:    addi sp, sp, 16
; RV32IFD-NEXT:    ret
;
; RV64IFD-LABEL: fcvt_lu_d:
; RV64IFD:       # %bb.0:
; RV64IFD-NEXT:    fmv.d.x ft0, a0
; RV64IFD-NEXT:    fcvt.lu.d a0, ft0, rtz
; RV64IFD-NEXT:    ret
;
; RV32I-LABEL: fcvt_lu_d:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -16
; RV32I-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32I-NEXT:    call __fixunsdfdi@plt
; RV32I-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32I-NEXT:    addi sp, sp, 16
; RV32I-NEXT:    ret
;
; RV64I-LABEL: fcvt_lu_d:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -16
; RV64I-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64I-NEXT:    call __fixunsdfdi@plt
; RV64I-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64I-NEXT:    addi sp, sp, 16
; RV64I-NEXT:    ret
  %1 = call i64 @llvm.experimental.constrained.fptoui.i64.f64(double %a, metadata !"fpexcept.strict") strictfp
  ret i64 %1
}
declare i64 @llvm.experimental.constrained.fptoui.i64.f64(double, metadata)

define double @fcvt_d_l(i64 %a) nounwind strictfp {
; RV32IFD-LABEL: fcvt_d_l:
; RV32IFD:       # %bb.0:
; RV32IFD-NEXT:    addi sp, sp, -16
; RV32IFD-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32IFD-NEXT:    call __floatdidf@plt
; RV32IFD-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32IFD-NEXT:    addi sp, sp, 16
; RV32IFD-NEXT:    ret
;
; RV64IFD-LABEL: fcvt_d_l:
; RV64IFD:       # %bb.0:
; RV64IFD-NEXT:    fcvt.d.l ft0, a0
; RV64IFD-NEXT:    fmv.x.d a0, ft0
; RV64IFD-NEXT:    ret
;
; RV32I-LABEL: fcvt_d_l:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -16
; RV32I-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32I-NEXT:    call __floatdidf@plt
; RV32I-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32I-NEXT:    addi sp, sp, 16
; RV32I-NEXT:    ret
;
; RV64I-LABEL: fcvt_d_l:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -16
; RV64I-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64I-NEXT:    call __floatdidf@plt
; RV64I-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64I-NEXT:    addi sp, sp, 16
; RV64I-NEXT:    ret
  %1 = call double @llvm.experimental.constrained.sitofp.f64.i64(i64 %a, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret double %1
}
declare double @llvm.experimental.constrained.sitofp.f64.i64(i64, metadata, metadata)

define double @fcvt_d_lu(i64 %a) nounwind strictfp {
; RV32IFD-LABEL: fcvt_d_lu:
; RV32IFD:       # %bb.0:
; RV32IFD-NEXT:    addi sp, sp, -16
; RV32IFD-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32IFD-NEXT:    call __floatundidf@plt
; RV32IFD-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32IFD-NEXT:    addi sp, sp, 16
; RV32IFD-NEXT:    ret
;
; RV64IFD-LABEL: fcvt_d_lu:
; RV64IFD:       # %bb.0:
; RV64IFD-NEXT:    fcvt.d.lu ft0, a0
; RV64IFD-NEXT:    fmv.x.d a0, ft0
; RV64IFD-NEXT:    ret
;
; RV32I-LABEL: fcvt_d_lu:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -16
; RV32I-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32I-NEXT:    call __floatundidf@plt
; RV32I-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32I-NEXT:    addi sp, sp, 16
; RV32I-NEXT:    ret
;
; RV64I-LABEL: fcvt_d_lu:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -16
; RV64I-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64I-NEXT:    call __floatundidf@plt
; RV64I-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64I-NEXT:    addi sp, sp, 16
; RV64I-NEXT:    ret
  %1 = call double @llvm.experimental.constrained.uitofp.f64.i64(i64 %a, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret double %1
}
declare double @llvm.experimental.constrained.uitofp.f64.i64(i64, metadata, metadata)

define double @fcvt_d_w_i8(i8 signext %a) nounwind strictfp {
; RV32IFD-LABEL: fcvt_d_w_i8:
; RV32IFD:       # %bb.0:
; RV32IFD-NEXT:    addi sp, sp, -16
; RV32IFD-NEXT:    fcvt.d.w ft0, a0
; RV32IFD-NEXT:    fsd ft0, 8(sp)
; RV32IFD-NEXT:    lw a0, 8(sp)
; RV32IFD-NEXT:    lw a1, 12(sp)
; RV32IFD-NEXT:    addi sp, sp, 16
; RV32IFD-NEXT:    ret
;
; RV64IFD-LABEL: fcvt_d_w_i8:
; RV64IFD:       # %bb.0:
; RV64IFD-NEXT:    fcvt.d.w ft0, a0
; RV64IFD-NEXT:    fmv.x.d a0, ft0
; RV64IFD-NEXT:    ret
;
; RV32I-LABEL: fcvt_d_w_i8:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -16
; RV32I-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32I-NEXT:    call __floatsidf@plt
; RV32I-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32I-NEXT:    addi sp, sp, 16
; RV32I-NEXT:    ret
;
; RV64I-LABEL: fcvt_d_w_i8:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -16
; RV64I-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64I-NEXT:    call __floatsidf@plt
; RV64I-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64I-NEXT:    addi sp, sp, 16
; RV64I-NEXT:    ret
  %1 = call double @llvm.experimental.constrained.sitofp.f64.i8(i8 %a, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret double %1
}
declare double @llvm.experimental.constrained.sitofp.f64.i8(i8, metadata, metadata)

define double @fcvt_d_wu_i8(i8 zeroext %a) nounwind strictfp {
; RV32IFD-LABEL: fcvt_d_wu_i8:
; RV32IFD:       # %bb.0:
; RV32IFD-NEXT:    addi sp, sp, -16
; RV32IFD-NEXT:    fcvt.d.wu ft0, a0
; RV32IFD-NEXT:    fsd ft0, 8(sp)
; RV32IFD-NEXT:    lw a0, 8(sp)
; RV32IFD-NEXT:    lw a1, 12(sp)
; RV32IFD-NEXT:    addi sp, sp, 16
; RV32IFD-NEXT:    ret
;
; RV64IFD-LABEL: fcvt_d_wu_i8:
; RV64IFD:       # %bb.0:
; RV64IFD-NEXT:    fcvt.d.wu ft0, a0
; RV64IFD-NEXT:    fmv.x.d a0, ft0
; RV64IFD-NEXT:    ret
;
; RV32I-LABEL: fcvt_d_wu_i8:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -16
; RV32I-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32I-NEXT:    call __floatunsidf@plt
; RV32I-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32I-NEXT:    addi sp, sp, 16
; RV32I-NEXT:    ret
;
; RV64I-LABEL: fcvt_d_wu_i8:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -16
; RV64I-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64I-NEXT:    call __floatunsidf@plt
; RV64I-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64I-NEXT:    addi sp, sp, 16
; RV64I-NEXT:    ret
  %1 = call double @llvm.experimental.constrained.uitofp.f64.i8(i8 %a, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret double %1
}
declare double @llvm.experimental.constrained.uitofp.f64.i8(i8, metadata, metadata)

define double @fcvt_d_w_i16(i16 signext %a) nounwind strictfp {
; RV32IFD-LABEL: fcvt_d_w_i16:
; RV32IFD:       # %bb.0:
; RV32IFD-NEXT:    addi sp, sp, -16
; RV32IFD-NEXT:    fcvt.d.w ft0, a0
; RV32IFD-NEXT:    fsd ft0, 8(sp)
; RV32IFD-NEXT:    lw a0, 8(sp)
; RV32IFD-NEXT:    lw a1, 12(sp)
; RV32IFD-NEXT:    addi sp, sp, 16
; RV32IFD-NEXT:    ret
;
; RV64IFD-LABEL: fcvt_d_w_i16:
; RV64IFD:       # %bb.0:
; RV64IFD-NEXT:    fcvt.d.w ft0, a0
; RV64IFD-NEXT:    fmv.x.d a0, ft0
; RV64IFD-NEXT:    ret
;
; RV32I-LABEL: fcvt_d_w_i16:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -16
; RV32I-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32I-NEXT:    call __floatsidf@plt
; RV32I-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32I-NEXT:    addi sp, sp, 16
; RV32I-NEXT:    ret
;
; RV64I-LABEL: fcvt_d_w_i16:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -16
; RV64I-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64I-NEXT:    call __floatsidf@plt
; RV64I-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64I-NEXT:    addi sp, sp, 16
; RV64I-NEXT:    ret
  %1 = call double @llvm.experimental.constrained.sitofp.f64.i16(i16 %a, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret double %1
}
declare double @llvm.experimental.constrained.sitofp.f64.i16(i16, metadata, metadata)

define double @fcvt_d_wu_i16(i16 zeroext %a) nounwind strictfp {
; RV32IFD-LABEL: fcvt_d_wu_i16:
; RV32IFD:       # %bb.0:
; RV32IFD-NEXT:    addi sp, sp, -16
; RV32IFD-NEXT:    fcvt.d.wu ft0, a0
; RV32IFD-NEXT:    fsd ft0, 8(sp)
; RV32IFD-NEXT:    lw a0, 8(sp)
; RV32IFD-NEXT:    lw a1, 12(sp)
; RV32IFD-NEXT:    addi sp, sp, 16
; RV32IFD-NEXT:    ret
;
; RV64IFD-LABEL: fcvt_d_wu_i16:
; RV64IFD:       # %bb.0:
; RV64IFD-NEXT:    fcvt.d.wu ft0, a0
; RV64IFD-NEXT:    fmv.x.d a0, ft0
; RV64IFD-NEXT:    ret
;
; RV32I-LABEL: fcvt_d_wu_i16:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -16
; RV32I-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32I-NEXT:    call __floatunsidf@plt
; RV32I-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32I-NEXT:    addi sp, sp, 16
; RV32I-NEXT:    ret
;
; RV64I-LABEL: fcvt_d_wu_i16:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -16
; RV64I-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64I-NEXT:    call __floatunsidf@plt
; RV64I-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64I-NEXT:    addi sp, sp, 16
; RV64I-NEXT:    ret
  %1 = call double @llvm.experimental.constrained.uitofp.f64.i16(i16 %a, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret double %1
}
declare double @llvm.experimental.constrained.uitofp.f64.i16(i16, metadata, metadata)

; Make sure we select W version of addi on RV64.
define signext i32 @fcvt_d_w_demanded_bits(i32 signext %0, double* %1) {
; RV32IFD-LABEL: fcvt_d_w_demanded_bits:
; RV32IFD:       # %bb.0:
; RV32IFD-NEXT:    addi a0, a0, 1
; RV32IFD-NEXT:    fcvt.d.w ft0, a0
; RV32IFD-NEXT:    fsd ft0, 0(a1)
; RV32IFD-NEXT:    ret
;
; RV64IFD-LABEL: fcvt_d_w_demanded_bits:
; RV64IFD:       # %bb.0:
; RV64IFD-NEXT:    addiw a0, a0, 1
; RV64IFD-NEXT:    fcvt.d.w ft0, a0
; RV64IFD-NEXT:    fsd ft0, 0(a1)
; RV64IFD-NEXT:    ret
;
; RV32I-LABEL: fcvt_d_w_demanded_bits:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -16
; RV32I-NEXT:    .cfi_def_cfa_offset 16
; RV32I-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32I-NEXT:    sw s0, 8(sp) # 4-byte Folded Spill
; RV32I-NEXT:    sw s1, 4(sp) # 4-byte Folded Spill
; RV32I-NEXT:    .cfi_offset ra, -4
; RV32I-NEXT:    .cfi_offset s0, -8
; RV32I-NEXT:    .cfi_offset s1, -12
; RV32I-NEXT:    mv s0, a1
; RV32I-NEXT:    addi s1, a0, 1
; RV32I-NEXT:    mv a0, s1
; RV32I-NEXT:    call __floatsidf@plt
; RV32I-NEXT:    sw a1, 4(s0)
; RV32I-NEXT:    sw a0, 0(s0)
; RV32I-NEXT:    mv a0, s1
; RV32I-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32I-NEXT:    lw s0, 8(sp) # 4-byte Folded Reload
; RV32I-NEXT:    lw s1, 4(sp) # 4-byte Folded Reload
; RV32I-NEXT:    addi sp, sp, 16
; RV32I-NEXT:    ret
;
; RV64I-LABEL: fcvt_d_w_demanded_bits:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -32
; RV64I-NEXT:    .cfi_def_cfa_offset 32
; RV64I-NEXT:    sd ra, 24(sp) # 8-byte Folded Spill
; RV64I-NEXT:    sd s0, 16(sp) # 8-byte Folded Spill
; RV64I-NEXT:    sd s1, 8(sp) # 8-byte Folded Spill
; RV64I-NEXT:    .cfi_offset ra, -8
; RV64I-NEXT:    .cfi_offset s0, -16
; RV64I-NEXT:    .cfi_offset s1, -24
; RV64I-NEXT:    mv s0, a1
; RV64I-NEXT:    addiw s1, a0, 1
; RV64I-NEXT:    mv a0, s1
; RV64I-NEXT:    call __floatsidf@plt
; RV64I-NEXT:    sd a0, 0(s0)
; RV64I-NEXT:    mv a0, s1
; RV64I-NEXT:    ld ra, 24(sp) # 8-byte Folded Reload
; RV64I-NEXT:    ld s0, 16(sp) # 8-byte Folded Reload
; RV64I-NEXT:    ld s1, 8(sp) # 8-byte Folded Reload
; RV64I-NEXT:    addi sp, sp, 32
; RV64I-NEXT:    ret
  %3 = add i32 %0, 1
  %4 = call double @llvm.experimental.constrained.sitofp.f64.i32(i32 %3, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  store double %4, double* %1, align 8
  ret i32 %3
}

; Make sure we select W version of addi on RV64.
define signext i32 @fcvt_d_wu_demanded_bits(i32 signext %0, double* %1) {
; RV32IFD-LABEL: fcvt_d_wu_demanded_bits:
; RV32IFD:       # %bb.0:
; RV32IFD-NEXT:    addi a0, a0, 1
; RV32IFD-NEXT:    fcvt.d.wu ft0, a0
; RV32IFD-NEXT:    fsd ft0, 0(a1)
; RV32IFD-NEXT:    ret
;
; RV64IFD-LABEL: fcvt_d_wu_demanded_bits:
; RV64IFD:       # %bb.0:
; RV64IFD-NEXT:    addiw a0, a0, 1
; RV64IFD-NEXT:    fcvt.d.wu ft0, a0
; RV64IFD-NEXT:    fsd ft0, 0(a1)
; RV64IFD-NEXT:    ret
;
; RV32I-LABEL: fcvt_d_wu_demanded_bits:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -16
; RV32I-NEXT:    .cfi_def_cfa_offset 16
; RV32I-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32I-NEXT:    sw s0, 8(sp) # 4-byte Folded Spill
; RV32I-NEXT:    sw s1, 4(sp) # 4-byte Folded Spill
; RV32I-NEXT:    .cfi_offset ra, -4
; RV32I-NEXT:    .cfi_offset s0, -8
; RV32I-NEXT:    .cfi_offset s1, -12
; RV32I-NEXT:    mv s0, a1
; RV32I-NEXT:    addi s1, a0, 1
; RV32I-NEXT:    mv a0, s1
; RV32I-NEXT:    call __floatunsidf@plt
; RV32I-NEXT:    sw a1, 4(s0)
; RV32I-NEXT:    sw a0, 0(s0)
; RV32I-NEXT:    mv a0, s1
; RV32I-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32I-NEXT:    lw s0, 8(sp) # 4-byte Folded Reload
; RV32I-NEXT:    lw s1, 4(sp) # 4-byte Folded Reload
; RV32I-NEXT:    addi sp, sp, 16
; RV32I-NEXT:    ret
;
; RV64I-LABEL: fcvt_d_wu_demanded_bits:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -32
; RV64I-NEXT:    .cfi_def_cfa_offset 32
; RV64I-NEXT:    sd ra, 24(sp) # 8-byte Folded Spill
; RV64I-NEXT:    sd s0, 16(sp) # 8-byte Folded Spill
; RV64I-NEXT:    sd s1, 8(sp) # 8-byte Folded Spill
; RV64I-NEXT:    .cfi_offset ra, -8
; RV64I-NEXT:    .cfi_offset s0, -16
; RV64I-NEXT:    .cfi_offset s1, -24
; RV64I-NEXT:    mv s0, a1
; RV64I-NEXT:    addiw s1, a0, 1
; RV64I-NEXT:    mv a0, s1
; RV64I-NEXT:    call __floatunsidf@plt
; RV64I-NEXT:    sd a0, 0(s0)
; RV64I-NEXT:    mv a0, s1
; RV64I-NEXT:    ld ra, 24(sp) # 8-byte Folded Reload
; RV64I-NEXT:    ld s0, 16(sp) # 8-byte Folded Reload
; RV64I-NEXT:    ld s1, 8(sp) # 8-byte Folded Reload
; RV64I-NEXT:    addi sp, sp, 32
; RV64I-NEXT:    ret
  %3 = add i32 %0, 1
  %4 = call double @llvm.experimental.constrained.uitofp.f64.i32(i32 %3, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  store double %4, double* %1, align 8
  ret i32 %3
}
