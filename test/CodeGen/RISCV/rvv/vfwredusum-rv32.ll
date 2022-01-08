; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv32 -mattr=+experimental-v,+d,+experimental-zfh -verify-machineinstrs \
; RUN:   < %s | FileCheck %s
declare <vscale x 2 x float> @llvm.riscv.vfwredusum.nxv2f32.nxv1f16(
  <vscale x 2 x float>,
  <vscale x 1 x half>,
  <vscale x 2 x float>,
  i32);

define <vscale x 2 x float> @intrinsic_vfwredusum_vs_nxv2f32_nxv1f16_nxv2f32(<vscale x 2 x float> %0, <vscale x 1 x half> %1, <vscale x 2 x float> %2, i32 %3) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_vs_nxv2f32_nxv1f16_nxv2f32:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e16, mf4, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v9, v10
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x float> @llvm.riscv.vfwredusum.nxv2f32.nxv1f16(
    <vscale x 2 x float> %0,
    <vscale x 1 x half> %1,
    <vscale x 2 x float> %2,
    i32 %3)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfwredusum.mask.nxv2f32.nxv1f16.nxv2f32(
  <vscale x 2 x float>,
  <vscale x 1 x half>,
  <vscale x 2 x float>,
  <vscale x 1 x i1>,
  i32);

define <vscale x 2 x float> @intrinsic_vfwredusum_mask_vs_nxv2f32_nxv1f16_nxv2f32(<vscale x 2 x float> %0, <vscale x 1 x half> %1, <vscale x 2 x float> %2, <vscale x 1 x i1> %3, i32 %4) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_mask_vs_nxv2f32_nxv1f16_nxv2f32:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e16, mf4, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v9, v10, v0.t
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x float> @llvm.riscv.vfwredusum.mask.nxv2f32.nxv1f16.nxv2f32(
    <vscale x 2 x float> %0,
    <vscale x 1 x half> %1,
    <vscale x 2 x float> %2,
    <vscale x 1 x i1> %3,
    i32 %4)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfwredusum.nxv2f32.nxv2f16(
  <vscale x 2 x float>,
  <vscale x 2 x half>,
  <vscale x 2 x float>,
  i32);

define <vscale x 2 x float> @intrinsic_vfwredusum_vs_nxv2f32_nxv2f16_nxv2f32(<vscale x 2 x float> %0, <vscale x 2 x half> %1, <vscale x 2 x float> %2, i32 %3) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_vs_nxv2f32_nxv2f16_nxv2f32:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e16, mf2, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v9, v10
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x float> @llvm.riscv.vfwredusum.nxv2f32.nxv2f16(
    <vscale x 2 x float> %0,
    <vscale x 2 x half> %1,
    <vscale x 2 x float> %2,
    i32 %3)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfwredusum.mask.nxv2f32.nxv2f16.nxv2f32(
  <vscale x 2 x float>,
  <vscale x 2 x half>,
  <vscale x 2 x float>,
  <vscale x 2 x i1>,
  i32);

define <vscale x 2 x float> @intrinsic_vfwredusum_mask_vs_nxv2f32_nxv2f16_nxv2f32(<vscale x 2 x float> %0, <vscale x 2 x half> %1, <vscale x 2 x float> %2, <vscale x 2 x i1> %3, i32 %4) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_mask_vs_nxv2f32_nxv2f16_nxv2f32:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e16, mf2, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v9, v10, v0.t
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x float> @llvm.riscv.vfwredusum.mask.nxv2f32.nxv2f16.nxv2f32(
    <vscale x 2 x float> %0,
    <vscale x 2 x half> %1,
    <vscale x 2 x float> %2,
    <vscale x 2 x i1> %3,
    i32 %4)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfwredusum.nxv2f32.nxv4f16(
  <vscale x 2 x float>,
  <vscale x 4 x half>,
  <vscale x 2 x float>,
  i32);

define <vscale x 2 x float> @intrinsic_vfwredusum_vs_nxv2f32_nxv4f16_nxv2f32(<vscale x 2 x float> %0, <vscale x 4 x half> %1, <vscale x 2 x float> %2, i32 %3) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_vs_nxv2f32_nxv4f16_nxv2f32:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e16, m1, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v9, v10
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x float> @llvm.riscv.vfwredusum.nxv2f32.nxv4f16(
    <vscale x 2 x float> %0,
    <vscale x 4 x half> %1,
    <vscale x 2 x float> %2,
    i32 %3)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfwredusum.mask.nxv2f32.nxv4f16.nxv2f32(
  <vscale x 2 x float>,
  <vscale x 4 x half>,
  <vscale x 2 x float>,
  <vscale x 4 x i1>,
  i32);

define <vscale x 2 x float> @intrinsic_vfwredusum_mask_vs_nxv2f32_nxv4f16_nxv2f32(<vscale x 2 x float> %0, <vscale x 4 x half> %1, <vscale x 2 x float> %2, <vscale x 4 x i1> %3, i32 %4) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_mask_vs_nxv2f32_nxv4f16_nxv2f32:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e16, m1, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v9, v10, v0.t
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x float> @llvm.riscv.vfwredusum.mask.nxv2f32.nxv4f16.nxv2f32(
    <vscale x 2 x float> %0,
    <vscale x 4 x half> %1,
    <vscale x 2 x float> %2,
    <vscale x 4 x i1> %3,
    i32 %4)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfwredusum.nxv2f32.nxv8f16(
  <vscale x 2 x float>,
  <vscale x 8 x half>,
  <vscale x 2 x float>,
  i32);

define <vscale x 2 x float> @intrinsic_vfwredusum_vs_nxv2f32_nxv8f16_nxv2f32(<vscale x 2 x float> %0, <vscale x 8 x half> %1, <vscale x 2 x float> %2, i32 %3) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_vs_nxv2f32_nxv8f16_nxv2f32:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e16, m2, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v10, v9
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x float> @llvm.riscv.vfwredusum.nxv2f32.nxv8f16(
    <vscale x 2 x float> %0,
    <vscale x 8 x half> %1,
    <vscale x 2 x float> %2,
    i32 %3)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfwredusum.mask.nxv2f32.nxv8f16.nxv2f32(
  <vscale x 2 x float>,
  <vscale x 8 x half>,
  <vscale x 2 x float>,
  <vscale x 8 x i1>,
  i32);

define <vscale x 2 x float> @intrinsic_vfwredusum_mask_vs_nxv2f32_nxv8f16_nxv2f32(<vscale x 2 x float> %0, <vscale x 8 x half> %1, <vscale x 2 x float> %2, <vscale x 8 x i1> %3, i32 %4) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_mask_vs_nxv2f32_nxv8f16_nxv2f32:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e16, m2, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v10, v9, v0.t
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x float> @llvm.riscv.vfwredusum.mask.nxv2f32.nxv8f16.nxv2f32(
    <vscale x 2 x float> %0,
    <vscale x 8 x half> %1,
    <vscale x 2 x float> %2,
    <vscale x 8 x i1> %3,
    i32 %4)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfwredusum.nxv2f32.nxv16f16(
  <vscale x 2 x float>,
  <vscale x 16 x half>,
  <vscale x 2 x float>,
  i32);

define <vscale x 2 x float> @intrinsic_vfwredusum_vs_nxv2f32_nxv16f16_nxv2f32(<vscale x 2 x float> %0, <vscale x 16 x half> %1, <vscale x 2 x float> %2, i32 %3) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_vs_nxv2f32_nxv16f16_nxv2f32:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e16, m4, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v12, v9
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x float> @llvm.riscv.vfwredusum.nxv2f32.nxv16f16(
    <vscale x 2 x float> %0,
    <vscale x 16 x half> %1,
    <vscale x 2 x float> %2,
    i32 %3)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfwredusum.mask.nxv2f32.nxv16f16.nxv2f32(
  <vscale x 2 x float>,
  <vscale x 16 x half>,
  <vscale x 2 x float>,
  <vscale x 16 x i1>,
  i32);

define <vscale x 2 x float> @intrinsic_vfwredusum_mask_vs_nxv2f32_nxv16f16_nxv2f32(<vscale x 2 x float> %0, <vscale x 16 x half> %1, <vscale x 2 x float> %2, <vscale x 16 x i1> %3, i32 %4) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_mask_vs_nxv2f32_nxv16f16_nxv2f32:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e16, m4, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v12, v9, v0.t
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x float> @llvm.riscv.vfwredusum.mask.nxv2f32.nxv16f16.nxv2f32(
    <vscale x 2 x float> %0,
    <vscale x 16 x half> %1,
    <vscale x 2 x float> %2,
    <vscale x 16 x i1> %3,
    i32 %4)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfwredusum.nxv2f32.nxv32f16(
  <vscale x 2 x float>,
  <vscale x 32 x half>,
  <vscale x 2 x float>,
  i32);

define <vscale x 2 x float> @intrinsic_vfwredusum_vs_nxv2f32_nxv32f16_nxv2f32(<vscale x 2 x float> %0, <vscale x 32 x half> %1, <vscale x 2 x float> %2, i32 %3) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_vs_nxv2f32_nxv32f16_nxv2f32:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e16, m8, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v16, v9
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x float> @llvm.riscv.vfwredusum.nxv2f32.nxv32f16(
    <vscale x 2 x float> %0,
    <vscale x 32 x half> %1,
    <vscale x 2 x float> %2,
    i32 %3)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfwredusum.mask.nxv2f32.nxv32f16(
  <vscale x 2 x float>,
  <vscale x 32 x half>,
  <vscale x 2 x float>,
  <vscale x 32 x i1>,
  i32);

define <vscale x 2 x float> @intrinsic_vfwredusum_mask_vs_nxv2f32_nxv32f16_nxv2f32(<vscale x 2 x float> %0, <vscale x 32 x half> %1, <vscale x 2 x float> %2, <vscale x 32 x i1> %3, i32 %4) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_mask_vs_nxv2f32_nxv32f16_nxv2f32:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e16, m8, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v16, v9, v0.t
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x float> @llvm.riscv.vfwredusum.mask.nxv2f32.nxv32f16(
    <vscale x 2 x float> %0,
    <vscale x 32 x half> %1,
    <vscale x 2 x float> %2,
    <vscale x 32 x i1> %3,
    i32 %4)

  ret <vscale x 2 x float> %a
}

declare <vscale x 1 x double> @llvm.riscv.vfwredusum.nxv1f64.nxv1f32(
  <vscale x 1 x double>,
  <vscale x 1 x float>,
  <vscale x 1 x double>,
  i32);

define <vscale x 1 x double> @intrinsic_vfwredusum_vs_nxv1f64_nxv1f32_nxv1f64(<vscale x 1 x double> %0, <vscale x 1 x float> %1, <vscale x 1 x double> %2, i32 %3) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_vs_nxv1f64_nxv1f32_nxv1f64:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, mf2, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v9, v10
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 1 x double> @llvm.riscv.vfwredusum.nxv1f64.nxv1f32(
    <vscale x 1 x double> %0,
    <vscale x 1 x float> %1,
    <vscale x 1 x double> %2,
    i32 %3)

  ret <vscale x 1 x double> %a
}

declare <vscale x 1 x double> @llvm.riscv.vfwredusum.mask.nxv1f64.nxv1f32.nxv1f64(
  <vscale x 1 x double>,
  <vscale x 1 x float>,
  <vscale x 1 x double>,
  <vscale x 1 x i1>,
  i32);

define <vscale x 1 x double> @intrinsic_vfwredusum_mask_vs_nxv1f64_nxv1f32_nxv1f64(<vscale x 1 x double> %0, <vscale x 1 x float> %1, <vscale x 1 x double> %2, <vscale x 1 x i1> %3, i32 %4) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_mask_vs_nxv1f64_nxv1f32_nxv1f64:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, mf2, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v9, v10, v0.t
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 1 x double> @llvm.riscv.vfwredusum.mask.nxv1f64.nxv1f32.nxv1f64(
    <vscale x 1 x double> %0,
    <vscale x 1 x float> %1,
    <vscale x 1 x double> %2,
    <vscale x 1 x i1> %3,
    i32 %4)

  ret <vscale x 1 x double> %a
}

declare <vscale x 1 x double> @llvm.riscv.vfwredusum.nxv1f64.nxv2f32(
  <vscale x 1 x double>,
  <vscale x 2 x float>,
  <vscale x 1 x double>,
  i32);

define <vscale x 1 x double> @intrinsic_vfwredusum_vs_nxv1f64_nxv2f32_nxv1f64(<vscale x 1 x double> %0, <vscale x 2 x float> %1, <vscale x 1 x double> %2, i32 %3) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_vs_nxv1f64_nxv2f32_nxv1f64:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v9, v10
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 1 x double> @llvm.riscv.vfwredusum.nxv1f64.nxv2f32(
    <vscale x 1 x double> %0,
    <vscale x 2 x float> %1,
    <vscale x 1 x double> %2,
    i32 %3)

  ret <vscale x 1 x double> %a
}

declare <vscale x 1 x double> @llvm.riscv.vfwredusum.mask.nxv1f64.nxv2f32.nxv1f64(
  <vscale x 1 x double>,
  <vscale x 2 x float>,
  <vscale x 1 x double>,
  <vscale x 2 x i1>,
  i32);

define <vscale x 1 x double> @intrinsic_vfwredusum_mask_vs_nxv1f64_nxv2f32_nxv1f64(<vscale x 1 x double> %0, <vscale x 2 x float> %1, <vscale x 1 x double> %2, <vscale x 2 x i1> %3, i32 %4) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_mask_vs_nxv1f64_nxv2f32_nxv1f64:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v9, v10, v0.t
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 1 x double> @llvm.riscv.vfwredusum.mask.nxv1f64.nxv2f32.nxv1f64(
    <vscale x 1 x double> %0,
    <vscale x 2 x float> %1,
    <vscale x 1 x double> %2,
    <vscale x 2 x i1> %3,
    i32 %4)

  ret <vscale x 1 x double> %a
}

declare <vscale x 1 x double> @llvm.riscv.vfwredusum.nxv1f64.nxv4f32(
  <vscale x 1 x double>,
  <vscale x 4 x float>,
  <vscale x 1 x double>,
  i32);

define <vscale x 1 x double> @intrinsic_vfwredusum_vs_nxv1f64_nxv4f32_nxv1f64(<vscale x 1 x double> %0, <vscale x 4 x float> %1, <vscale x 1 x double> %2, i32 %3) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_vs_nxv1f64_nxv4f32_nxv1f64:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m2, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v10, v9
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 1 x double> @llvm.riscv.vfwredusum.nxv1f64.nxv4f32(
    <vscale x 1 x double> %0,
    <vscale x 4 x float> %1,
    <vscale x 1 x double> %2,
    i32 %3)

  ret <vscale x 1 x double> %a
}

declare <vscale x 1 x double> @llvm.riscv.vfwredusum.mask.nxv1f64.nxv4f32.nxv1f64(
  <vscale x 1 x double>,
  <vscale x 4 x float>,
  <vscale x 1 x double>,
  <vscale x 4 x i1>,
  i32);

define <vscale x 1 x double> @intrinsic_vfwredusum_mask_vs_nxv1f64_nxv4f32_nxv1f64(<vscale x 1 x double> %0, <vscale x 4 x float> %1, <vscale x 1 x double> %2, <vscale x 4 x i1> %3, i32 %4) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_mask_vs_nxv1f64_nxv4f32_nxv1f64:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m2, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v10, v9, v0.t
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 1 x double> @llvm.riscv.vfwredusum.mask.nxv1f64.nxv4f32.nxv1f64(
    <vscale x 1 x double> %0,
    <vscale x 4 x float> %1,
    <vscale x 1 x double> %2,
    <vscale x 4 x i1> %3,
    i32 %4)

  ret <vscale x 1 x double> %a
}

declare <vscale x 1 x double> @llvm.riscv.vfwredusum.nxv1f64.nxv8f32(
  <vscale x 1 x double>,
  <vscale x 8 x float>,
  <vscale x 1 x double>,
  i32);

define <vscale x 1 x double> @intrinsic_vfwredusum_vs_nxv1f64_nxv8f32_nxv1f64(<vscale x 1 x double> %0, <vscale x 8 x float> %1, <vscale x 1 x double> %2, i32 %3) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_vs_nxv1f64_nxv8f32_nxv1f64:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m4, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v12, v9
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 1 x double> @llvm.riscv.vfwredusum.nxv1f64.nxv8f32(
    <vscale x 1 x double> %0,
    <vscale x 8 x float> %1,
    <vscale x 1 x double> %2,
    i32 %3)

  ret <vscale x 1 x double> %a
}

declare <vscale x 1 x double> @llvm.riscv.vfwredusum.mask.nxv1f64.nxv8f32.nxv1f64(
  <vscale x 1 x double>,
  <vscale x 8 x float>,
  <vscale x 1 x double>,
  <vscale x 8 x i1>,
  i32);

define <vscale x 1 x double> @intrinsic_vfwredusum_mask_vs_nxv1f64_nxv8f32_nxv1f64(<vscale x 1 x double> %0, <vscale x 8 x float> %1, <vscale x 1 x double> %2, <vscale x 8 x i1> %3, i32 %4) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_mask_vs_nxv1f64_nxv8f32_nxv1f64:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m4, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v12, v9, v0.t
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 1 x double> @llvm.riscv.vfwredusum.mask.nxv1f64.nxv8f32.nxv1f64(
    <vscale x 1 x double> %0,
    <vscale x 8 x float> %1,
    <vscale x 1 x double> %2,
    <vscale x 8 x i1> %3,
    i32 %4)

  ret <vscale x 1 x double> %a
}

declare <vscale x 1 x double> @llvm.riscv.vfwredusum.nxv1f64.nxv16f32(
  <vscale x 1 x double>,
  <vscale x 16 x float>,
  <vscale x 1 x double>,
  i32);

define <vscale x 1 x double> @intrinsic_vfwredusum_vs_nxv1f64_nxv16f32_nxv1f64(<vscale x 1 x double> %0, <vscale x 16 x float> %1, <vscale x 1 x double> %2, i32 %3) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_vs_nxv1f64_nxv16f32_nxv1f64:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m8, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v16, v9
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 1 x double> @llvm.riscv.vfwredusum.nxv1f64.nxv16f32(
    <vscale x 1 x double> %0,
    <vscale x 16 x float> %1,
    <vscale x 1 x double> %2,
    i32 %3)

  ret <vscale x 1 x double> %a
}

declare <vscale x 1 x double> @llvm.riscv.vfwredusum.mask.nxv1f64.nxv16f32.nxv1f64(
  <vscale x 1 x double>,
  <vscale x 16 x float>,
  <vscale x 1 x double>,
  <vscale x 16 x i1>,
  i32);

define <vscale x 1 x double> @intrinsic_vfwredusum_mask_vs_nxv1f64_nxv16f32_nxv1f64(<vscale x 1 x double> %0, <vscale x 16 x float> %1, <vscale x 1 x double> %2, <vscale x 16 x i1> %3, i32 %4) nounwind {
; CHECK-LABEL: intrinsic_vfwredusum_mask_vs_nxv1f64_nxv16f32_nxv1f64:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m8, tu, mu
; CHECK-NEXT:    vfwredusum.vs v8, v16, v9, v0.t
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 1 x double> @llvm.riscv.vfwredusum.mask.nxv1f64.nxv16f32.nxv1f64(
    <vscale x 1 x double> %0,
    <vscale x 16 x float> %1,
    <vscale x 1 x double> %2,
    <vscale x 16 x i1> %3,
    i32 %4)

  ret <vscale x 1 x double> %a
}
