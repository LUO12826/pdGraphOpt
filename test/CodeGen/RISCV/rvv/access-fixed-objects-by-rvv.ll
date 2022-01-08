; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+m,+experimental-v -O2 < %s \
; RUN:   | FileCheck %s -check-prefix=RV64IV

define <vscale x 1 x i64> @access_fixed_object(i64 *%val) {
; RV64IV-LABEL: access_fixed_object:
; RV64IV:       # %bb.0:
; RV64IV-NEXT:    addi sp, sp, -528
; RV64IV-NEXT:    .cfi_def_cfa_offset 528
; RV64IV-NEXT:    addi a1, sp, 8
; RV64IV-NEXT:    vl1re64.v v8, (a1)
; RV64IV-NEXT:    ld a1, 520(sp)
; RV64IV-NEXT:    sd a1, 0(a0)
; RV64IV-NEXT:    addi sp, sp, 528
; RV64IV-NEXT:    ret
  %local = alloca i64
  %array = alloca [64 x i64]
  %vptr = bitcast [64 x i64]* %array to <vscale x 1 x i64>*
  %v = load <vscale x 1 x i64>, <vscale x 1 x i64>* %vptr
  %len = load i64, i64* %local
  store i64 %len, i64* %val
  ret <vscale x 1 x i64> %v
}

declare <vscale x 1 x i64> @llvm.riscv.vadd.nxv1i64.nxv1i64(
  <vscale x 1 x i64>,
  <vscale x 1 x i64>,
  i64);

define <vscale x 1 x i64> @access_fixed_and_vector_objects(i64 *%val) {
; RV64IV-LABEL: access_fixed_and_vector_objects:
; RV64IV:       # %bb.0:
; RV64IV-NEXT:    addi sp, sp, -544
; RV64IV-NEXT:    .cfi_def_cfa_offset 544
; RV64IV-NEXT:    csrr a0, vlenb
; RV64IV-NEXT:    sub sp, sp, a0
; RV64IV-NEXT:    addi a0, sp, 24
; RV64IV-NEXT:    vl1re64.v v8, (a0)
; RV64IV-NEXT:    ld a0, 536(sp)
; RV64IV-NEXT:    addi a1, sp, 544
; RV64IV-NEXT:    vl1re64.v v9, (a1)
; RV64IV-NEXT:    vsetvli zero, a0, e64, m1, ta, mu
; RV64IV-NEXT:    vadd.vv v8, v8, v9
; RV64IV-NEXT:    csrr a0, vlenb
; RV64IV-NEXT:    add sp, sp, a0
; RV64IV-NEXT:    addi sp, sp, 544
; RV64IV-NEXT:    ret
  %local = alloca i64
  %vector = alloca <vscale x 1 x i64>
  %array = alloca [64 x i64]
  %vptr = bitcast [64 x i64]* %array to <vscale x 1 x i64>*
  %v1 = load <vscale x 1 x i64>, <vscale x 1 x i64>* %vptr
  %v2 = load <vscale x 1 x i64>, <vscale x 1 x i64>* %vector
  %len = load i64, i64* %local

  %a = call <vscale x 1 x i64> @llvm.riscv.vadd.nxv1i64.nxv1i64(
    <vscale x 1 x i64> %v1,
    <vscale x 1 x i64> %v2,
    i64 %len)

  ret <vscale x 1 x i64> %a
}
