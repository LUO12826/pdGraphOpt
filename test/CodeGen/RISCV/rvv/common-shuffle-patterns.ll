; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+experimental-v -verify-machineinstrs -riscv-v-vector-bits-min=128 < %s | FileCheck %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown-elf"

define dso_local <16 x i16> @interleave(<8 x i16> %v0, <8 x i16> %v1) {
; CHECK-LABEL: interleave:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vmv1r.v v10, v9
; CHECK-NEXT:    # kill: def $v8 killed $v8 def $v8m2
; CHECK-NEXT:    vsetivli zero, 16, e16, m2, ta, mu
; CHECK-NEXT:    vmv.v.i v12, 0
; CHECK-NEXT:    vsetivli zero, 8, e16, m2, tu, mu
; CHECK-NEXT:    vmv2r.v v14, v12
; CHECK-NEXT:    vslideup.vi v14, v8, 0
; CHECK-NEXT:    vsetivli zero, 8, e16, m1, ta, mu
; CHECK-NEXT:    vmv.v.i v8, 0
; CHECK-NEXT:    vsetivli zero, 16, e16, m2, tu, mu
; CHECK-NEXT:    vslideup.vi v14, v8, 8
; CHECK-NEXT:    vsetvli zero, zero, e16, m2, ta, mu
; CHECK-NEXT:    vid.v v16
; CHECK-NEXT:    vsrl.vi v18, v16, 1
; CHECK-NEXT:    vrgather.vv v20, v14, v18
; CHECK-NEXT:    vsetivli zero, 8, e16, m2, tu, mu
; CHECK-NEXT:    vslideup.vi v12, v10, 0
; CHECK-NEXT:    vsetivli zero, 16, e16, m2, tu, mu
; CHECK-NEXT:    vslideup.vi v12, v8, 8
; CHECK-NEXT:    vsetvli zero, zero, e16, m2, ta, mu
; CHECK-NEXT:    lui a0, 11
; CHECK-NEXT:    addiw a0, a0, -1366
; CHECK-NEXT:    vmv.s.x v0, a0
; CHECK-NEXT:    vrgather.vv v8, v20, v16
; CHECK-NEXT:    vrgather.vv v8, v12, v18, v0.t
; CHECK-NEXT:    ret
entry:
  %v2 = shufflevector <8 x i16> %v0, <8 x i16> poison, <16 x i32> <i32 0, i32 undef, i32 1, i32 undef, i32 2, i32 undef, i32 3, i32 undef, i32 4, i32 undef, i32 5, i32 undef, i32 6, i32 undef, i32 7, i32 undef>
  %v3 = shufflevector <8 x i16> %v1, <8 x i16> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %v4 = shufflevector <16 x i16> %v2, <16 x i16> %v3, <16 x i32> <i32 0, i32 16, i32 2, i32 17, i32 4, i32 18, i32 6, i32 19, i32 8, i32 20, i32 10, i32 21, i32 12, i32 22, i32 14, i32 23>
  ret <16 x i16> %v4
}
