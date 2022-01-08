; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv32 -mattr=+experimental-v -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,RV32
; RUN: llc -mtriple=riscv64 -mattr=+experimental-v -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,RV64

define <vscale x 1 x i32> @vmulhu_vv_nxv1i32(<vscale x 1 x i32> %va, <vscale x 1 x i32> %vb) {
; CHECK-LABEL: vmulhu_vv_nxv1i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a0, zero, e32, mf2, ta, mu
; CHECK-NEXT:    vmulhu.vv v8, v9, v8
; CHECK-NEXT:    ret
  %vc = zext <vscale x 1 x i32> %vb to <vscale x 1 x i64>
  %vd = zext <vscale x 1 x i32> %va to <vscale x 1 x i64>
  %ve = mul <vscale x 1 x i64> %vc, %vd
  %head = insertelement <vscale x 1 x i64> undef, i64 32, i32 0
  %splat = shufflevector <vscale x 1 x i64> %head, <vscale x 1 x i64> undef, <vscale x 1 x i32> zeroinitializer
  %vf = lshr <vscale x 1 x i64> %ve, %splat
  %vg = trunc <vscale x 1 x i64> %vf to <vscale x 1 x i32>
  ret <vscale x 1 x i32> %vg
}

define <vscale x 1 x i32> @vmulhu_vx_nxv1i32(<vscale x 1 x i32> %va, i32 %x) {
; CHECK-LABEL: vmulhu_vx_nxv1i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e32, mf2, ta, mu
; CHECK-NEXT:    vmulhu.vx v8, v8, a0
; CHECK-NEXT:    ret
  %head1 = insertelement <vscale x 1 x i32> undef, i32 %x, i32 0
  %splat1 = shufflevector <vscale x 1 x i32> %head1, <vscale x 1 x i32> undef, <vscale x 1 x i32> zeroinitializer
  %vb = zext <vscale x 1 x i32> %splat1 to <vscale x 1 x i64>
  %vc = zext <vscale x 1 x i32> %va to <vscale x 1 x i64>
  %vd = mul <vscale x 1 x i64> %vb, %vc
  %head2 = insertelement <vscale x 1 x i64> undef, i64 32, i32 0
  %splat2 = shufflevector <vscale x 1 x i64> %head2, <vscale x 1 x i64> undef, <vscale x 1 x i32> zeroinitializer
  %ve = lshr <vscale x 1 x i64> %vd, %splat2
  %vf = trunc <vscale x 1 x i64> %ve to <vscale x 1 x i32>
  ret <vscale x 1 x i32> %vf
}

define <vscale x 1 x i32> @vmulhu_vi_nxv1i32_0(<vscale x 1 x i32> %va) {
; RV32-LABEL: vmulhu_vi_nxv1i32_0:
; RV32:       # %bb.0:
; RV32-NEXT:    li a0, -7
; RV32-NEXT:    vsetvli a1, zero, e32, mf2, ta, mu
; RV32-NEXT:    vmulhu.vx v8, v8, a0
; RV32-NEXT:    ret
;
; RV64-LABEL: vmulhu_vi_nxv1i32_0:
; RV64:       # %bb.0:
; RV64-NEXT:    li a0, 1
; RV64-NEXT:    slli a0, a0, 32
; RV64-NEXT:    addi a0, a0, -7
; RV64-NEXT:    vsetvli a1, zero, e32, mf2, ta, mu
; RV64-NEXT:    vmulhu.vx v8, v8, a0
; RV64-NEXT:    ret
  %head1 = insertelement <vscale x 1 x i32> undef, i32 -7, i32 0
  %splat1 = shufflevector <vscale x 1 x i32> %head1, <vscale x 1 x i32> undef, <vscale x 1 x i32> zeroinitializer
  %vb = zext <vscale x 1 x i32> %splat1 to <vscale x 1 x i64>
  %vc = zext <vscale x 1 x i32> %va to <vscale x 1 x i64>
  %vd = mul <vscale x 1 x i64> %vb, %vc
  %head2 = insertelement <vscale x 1 x i64> undef, i64 32, i32 0
  %splat2 = shufflevector <vscale x 1 x i64> %head2, <vscale x 1 x i64> undef, <vscale x 1 x i32> zeroinitializer
  %ve = lshr <vscale x 1 x i64> %vd, %splat2
  %vf = trunc <vscale x 1 x i64> %ve to <vscale x 1 x i32>
  ret <vscale x 1 x i32> %vf
}

define <vscale x 1 x i32> @vmulhu_vi_nxv1i32_1(<vscale x 1 x i32> %va) {
; RV32-LABEL: vmulhu_vi_nxv1i32_1:
; RV32:       # %bb.0:
; RV32-NEXT:    vsetvli a0, zero, e32, mf2, ta, mu
; RV32-NEXT:    vsrl.vi v8, v8, 28
; RV32-NEXT:    ret
;
; RV64-LABEL: vmulhu_vi_nxv1i32_1:
; RV64:       # %bb.0:
; RV64-NEXT:    li a0, 16
; RV64-NEXT:    vsetvli a1, zero, e32, mf2, ta, mu
; RV64-NEXT:    vmulhu.vx v8, v8, a0
; RV64-NEXT:    ret
  %head1 = insertelement <vscale x 1 x i32> undef, i32 16, i32 0
  %splat1 = shufflevector <vscale x 1 x i32> %head1, <vscale x 1 x i32> undef, <vscale x 1 x i32> zeroinitializer
  %vb = zext <vscale x 1 x i32> %splat1 to <vscale x 1 x i64>
  %vc = zext <vscale x 1 x i32> %va to <vscale x 1 x i64>
  %vd = mul <vscale x 1 x i64> %vb, %vc
  %head2 = insertelement <vscale x 1 x i64> undef, i64 32, i32 0
  %splat2 = shufflevector <vscale x 1 x i64> %head2, <vscale x 1 x i64> undef, <vscale x 1 x i32> zeroinitializer
  %ve = lshr <vscale x 1 x i64> %vd, %splat2
  %vf = trunc <vscale x 1 x i64> %ve to <vscale x 1 x i32>
  ret <vscale x 1 x i32> %vf
}

define <vscale x 2 x i32> @vmulhu_vv_nxv2i32(<vscale x 2 x i32> %va, <vscale x 2 x i32> %vb) {
; CHECK-LABEL: vmulhu_vv_nxv2i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a0, zero, e32, m1, ta, mu
; CHECK-NEXT:    vmulhu.vv v8, v9, v8
; CHECK-NEXT:    ret
  %vc = zext <vscale x 2 x i32> %vb to <vscale x 2 x i64>
  %vd = zext <vscale x 2 x i32> %va to <vscale x 2 x i64>
  %ve = mul <vscale x 2 x i64> %vc, %vd
  %head = insertelement <vscale x 2 x i64> undef, i64 32, i32 0
  %splat = shufflevector <vscale x 2 x i64> %head, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %vf = lshr <vscale x 2 x i64> %ve, %splat
  %vg = trunc <vscale x 2 x i64> %vf to <vscale x 2 x i32>
  ret <vscale x 2 x i32> %vg
}

define <vscale x 2 x i32> @vmulhu_vx_nxv2i32(<vscale x 2 x i32> %va, i32 %x) {
; CHECK-LABEL: vmulhu_vx_nxv2i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e32, m1, ta, mu
; CHECK-NEXT:    vmulhu.vx v8, v8, a0
; CHECK-NEXT:    ret
  %head1 = insertelement <vscale x 2 x i32> undef, i32 %x, i32 0
  %splat1 = shufflevector <vscale x 2 x i32> %head1, <vscale x 2 x i32> undef, <vscale x 2 x i32> zeroinitializer
  %vb = zext <vscale x 2 x i32> %splat1 to <vscale x 2 x i64>
  %vc = zext <vscale x 2 x i32> %va to <vscale x 2 x i64>
  %vd = mul <vscale x 2 x i64> %vb, %vc
  %head2 = insertelement <vscale x 2 x i64> undef, i64 32, i32 0
  %splat2 = shufflevector <vscale x 2 x i64> %head2, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %ve = lshr <vscale x 2 x i64> %vd, %splat2
  %vf = trunc <vscale x 2 x i64> %ve to <vscale x 2 x i32>
  ret <vscale x 2 x i32> %vf
}

define <vscale x 2 x i32> @vmulhu_vi_nxv2i32_0(<vscale x 2 x i32> %va) {
; RV32-LABEL: vmulhu_vi_nxv2i32_0:
; RV32:       # %bb.0:
; RV32-NEXT:    li a0, -7
; RV32-NEXT:    vsetvli a1, zero, e32, m1, ta, mu
; RV32-NEXT:    vmulhu.vx v8, v8, a0
; RV32-NEXT:    ret
;
; RV64-LABEL: vmulhu_vi_nxv2i32_0:
; RV64:       # %bb.0:
; RV64-NEXT:    li a0, 1
; RV64-NEXT:    slli a0, a0, 32
; RV64-NEXT:    addi a0, a0, -7
; RV64-NEXT:    vsetvli a1, zero, e32, m1, ta, mu
; RV64-NEXT:    vmulhu.vx v8, v8, a0
; RV64-NEXT:    ret
  %head1 = insertelement <vscale x 2 x i32> undef, i32 -7, i32 0
  %splat1 = shufflevector <vscale x 2 x i32> %head1, <vscale x 2 x i32> undef, <vscale x 2 x i32> zeroinitializer
  %vb = zext <vscale x 2 x i32> %splat1 to <vscale x 2 x i64>
  %vc = zext <vscale x 2 x i32> %va to <vscale x 2 x i64>
  %vd = mul <vscale x 2 x i64> %vb, %vc
  %head2 = insertelement <vscale x 2 x i64> undef, i64 32, i32 0
  %splat2 = shufflevector <vscale x 2 x i64> %head2, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %ve = lshr <vscale x 2 x i64> %vd, %splat2
  %vf = trunc <vscale x 2 x i64> %ve to <vscale x 2 x i32>
  ret <vscale x 2 x i32> %vf
}

define <vscale x 2 x i32> @vmulhu_vi_nxv2i32_1(<vscale x 2 x i32> %va) {
; RV32-LABEL: vmulhu_vi_nxv2i32_1:
; RV32:       # %bb.0:
; RV32-NEXT:    vsetvli a0, zero, e32, m1, ta, mu
; RV32-NEXT:    vsrl.vi v8, v8, 28
; RV32-NEXT:    ret
;
; RV64-LABEL: vmulhu_vi_nxv2i32_1:
; RV64:       # %bb.0:
; RV64-NEXT:    li a0, 16
; RV64-NEXT:    vsetvli a1, zero, e32, m1, ta, mu
; RV64-NEXT:    vmulhu.vx v8, v8, a0
; RV64-NEXT:    ret
  %head1 = insertelement <vscale x 2 x i32> undef, i32 16, i32 0
  %splat1 = shufflevector <vscale x 2 x i32> %head1, <vscale x 2 x i32> undef, <vscale x 2 x i32> zeroinitializer
  %vb = zext <vscale x 2 x i32> %splat1 to <vscale x 2 x i64>
  %vc = zext <vscale x 2 x i32> %va to <vscale x 2 x i64>
  %vd = mul <vscale x 2 x i64> %vb, %vc
  %head2 = insertelement <vscale x 2 x i64> undef, i64 32, i32 0
  %splat2 = shufflevector <vscale x 2 x i64> %head2, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %ve = lshr <vscale x 2 x i64> %vd, %splat2
  %vf = trunc <vscale x 2 x i64> %ve to <vscale x 2 x i32>
  ret <vscale x 2 x i32> %vf
}

define <vscale x 4 x i32> @vmulhu_vv_nxv4i32(<vscale x 4 x i32> %va, <vscale x 4 x i32> %vb) {
; CHECK-LABEL: vmulhu_vv_nxv4i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a0, zero, e32, m2, ta, mu
; CHECK-NEXT:    vmulhu.vv v8, v10, v8
; CHECK-NEXT:    ret
  %vc = zext <vscale x 4 x i32> %vb to <vscale x 4 x i64>
  %vd = zext <vscale x 4 x i32> %va to <vscale x 4 x i64>
  %ve = mul <vscale x 4 x i64> %vc, %vd
  %head = insertelement <vscale x 4 x i64> undef, i64 32, i32 0
  %splat = shufflevector <vscale x 4 x i64> %head, <vscale x 4 x i64> undef, <vscale x 4 x i32> zeroinitializer
  %vf = lshr <vscale x 4 x i64> %ve, %splat
  %vg = trunc <vscale x 4 x i64> %vf to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %vg
}

define <vscale x 4 x i32> @vmulhu_vx_nxv4i32(<vscale x 4 x i32> %va, i32 %x) {
; CHECK-LABEL: vmulhu_vx_nxv4i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e32, m2, ta, mu
; CHECK-NEXT:    vmulhu.vx v8, v8, a0
; CHECK-NEXT:    ret
  %head1 = insertelement <vscale x 4 x i32> undef, i32 %x, i32 0
  %splat1 = shufflevector <vscale x 4 x i32> %head1, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %vb = zext <vscale x 4 x i32> %splat1 to <vscale x 4 x i64>
  %vc = zext <vscale x 4 x i32> %va to <vscale x 4 x i64>
  %vd = mul <vscale x 4 x i64> %vb, %vc
  %head2 = insertelement <vscale x 4 x i64> undef, i64 32, i32 0
  %splat2 = shufflevector <vscale x 4 x i64> %head2, <vscale x 4 x i64> undef, <vscale x 4 x i32> zeroinitializer
  %ve = lshr <vscale x 4 x i64> %vd, %splat2
  %vf = trunc <vscale x 4 x i64> %ve to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %vf
}

define <vscale x 4 x i32> @vmulhu_vi_nxv4i32_0(<vscale x 4 x i32> %va) {
; RV32-LABEL: vmulhu_vi_nxv4i32_0:
; RV32:       # %bb.0:
; RV32-NEXT:    li a0, -7
; RV32-NEXT:    vsetvli a1, zero, e32, m2, ta, mu
; RV32-NEXT:    vmulhu.vx v8, v8, a0
; RV32-NEXT:    ret
;
; RV64-LABEL: vmulhu_vi_nxv4i32_0:
; RV64:       # %bb.0:
; RV64-NEXT:    li a0, 1
; RV64-NEXT:    slli a0, a0, 32
; RV64-NEXT:    addi a0, a0, -7
; RV64-NEXT:    vsetvli a1, zero, e32, m2, ta, mu
; RV64-NEXT:    vmulhu.vx v8, v8, a0
; RV64-NEXT:    ret
  %head1 = insertelement <vscale x 4 x i32> undef, i32 -7, i32 0
  %splat1 = shufflevector <vscale x 4 x i32> %head1, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %vb = zext <vscale x 4 x i32> %splat1 to <vscale x 4 x i64>
  %vc = zext <vscale x 4 x i32> %va to <vscale x 4 x i64>
  %vd = mul <vscale x 4 x i64> %vb, %vc
  %head2 = insertelement <vscale x 4 x i64> undef, i64 32, i32 0
  %splat2 = shufflevector <vscale x 4 x i64> %head2, <vscale x 4 x i64> undef, <vscale x 4 x i32> zeroinitializer
  %ve = lshr <vscale x 4 x i64> %vd, %splat2
  %vf = trunc <vscale x 4 x i64> %ve to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %vf
}

define <vscale x 4 x i32> @vmulhu_vi_nxv4i32_1(<vscale x 4 x i32> %va) {
; RV32-LABEL: vmulhu_vi_nxv4i32_1:
; RV32:       # %bb.0:
; RV32-NEXT:    vsetvli a0, zero, e32, m2, ta, mu
; RV32-NEXT:    vsrl.vi v8, v8, 28
; RV32-NEXT:    ret
;
; RV64-LABEL: vmulhu_vi_nxv4i32_1:
; RV64:       # %bb.0:
; RV64-NEXT:    li a0, 16
; RV64-NEXT:    vsetvli a1, zero, e32, m2, ta, mu
; RV64-NEXT:    vmulhu.vx v8, v8, a0
; RV64-NEXT:    ret
  %head1 = insertelement <vscale x 4 x i32> undef, i32 16, i32 0
  %splat1 = shufflevector <vscale x 4 x i32> %head1, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %vb = zext <vscale x 4 x i32> %splat1 to <vscale x 4 x i64>
  %vc = zext <vscale x 4 x i32> %va to <vscale x 4 x i64>
  %vd = mul <vscale x 4 x i64> %vb, %vc
  %head2 = insertelement <vscale x 4 x i64> undef, i64 32, i32 0
  %splat2 = shufflevector <vscale x 4 x i64> %head2, <vscale x 4 x i64> undef, <vscale x 4 x i32> zeroinitializer
  %ve = lshr <vscale x 4 x i64> %vd, %splat2
  %vf = trunc <vscale x 4 x i64> %ve to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %vf
}

define <vscale x 8 x i32> @vmulhu_vv_nxv8i32(<vscale x 8 x i32> %va, <vscale x 8 x i32> %vb) {
; CHECK-LABEL: vmulhu_vv_nxv8i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a0, zero, e32, m4, ta, mu
; CHECK-NEXT:    vmulhu.vv v8, v12, v8
; CHECK-NEXT:    ret
  %vc = zext <vscale x 8 x i32> %vb to <vscale x 8 x i64>
  %vd = zext <vscale x 8 x i32> %va to <vscale x 8 x i64>
  %ve = mul <vscale x 8 x i64> %vc, %vd
  %head = insertelement <vscale x 8 x i64> undef, i64 32, i32 0
  %splat = shufflevector <vscale x 8 x i64> %head, <vscale x 8 x i64> undef, <vscale x 8 x i32> zeroinitializer
  %vf = lshr <vscale x 8 x i64> %ve, %splat
  %vg = trunc <vscale x 8 x i64> %vf to <vscale x 8 x i32>
  ret <vscale x 8 x i32> %vg
}

define <vscale x 8 x i32> @vmulhu_vx_nxv8i32(<vscale x 8 x i32> %va, i32 %x) {
; CHECK-LABEL: vmulhu_vx_nxv8i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e32, m4, ta, mu
; CHECK-NEXT:    vmulhu.vx v8, v8, a0
; CHECK-NEXT:    ret
  %head1 = insertelement <vscale x 8 x i32> undef, i32 %x, i32 0
  %splat1 = shufflevector <vscale x 8 x i32> %head1, <vscale x 8 x i32> undef, <vscale x 8 x i32> zeroinitializer
  %vb = zext <vscale x 8 x i32> %splat1 to <vscale x 8 x i64>
  %vc = zext <vscale x 8 x i32> %va to <vscale x 8 x i64>
  %vd = mul <vscale x 8 x i64> %vb, %vc
  %head2 = insertelement <vscale x 8 x i64> undef, i64 32, i32 0
  %splat2 = shufflevector <vscale x 8 x i64> %head2, <vscale x 8 x i64> undef, <vscale x 8 x i32> zeroinitializer
  %ve = lshr <vscale x 8 x i64> %vd, %splat2
  %vf = trunc <vscale x 8 x i64> %ve to <vscale x 8 x i32>
  ret <vscale x 8 x i32> %vf
}

define <vscale x 8 x i32> @vmulhu_vi_nxv8i32_0(<vscale x 8 x i32> %va) {
; RV32-LABEL: vmulhu_vi_nxv8i32_0:
; RV32:       # %bb.0:
; RV32-NEXT:    li a0, -7
; RV32-NEXT:    vsetvli a1, zero, e32, m4, ta, mu
; RV32-NEXT:    vmulhu.vx v8, v8, a0
; RV32-NEXT:    ret
;
; RV64-LABEL: vmulhu_vi_nxv8i32_0:
; RV64:       # %bb.0:
; RV64-NEXT:    li a0, 1
; RV64-NEXT:    slli a0, a0, 32
; RV64-NEXT:    addi a0, a0, -7
; RV64-NEXT:    vsetvli a1, zero, e32, m4, ta, mu
; RV64-NEXT:    vmulhu.vx v8, v8, a0
; RV64-NEXT:    ret
  %head1 = insertelement <vscale x 8 x i32> undef, i32 -7, i32 0
  %splat1 = shufflevector <vscale x 8 x i32> %head1, <vscale x 8 x i32> undef, <vscale x 8 x i32> zeroinitializer
  %vb = zext <vscale x 8 x i32> %splat1 to <vscale x 8 x i64>
  %vc = zext <vscale x 8 x i32> %va to <vscale x 8 x i64>
  %vd = mul <vscale x 8 x i64> %vb, %vc
  %head2 = insertelement <vscale x 8 x i64> undef, i64 32, i32 0
  %splat2 = shufflevector <vscale x 8 x i64> %head2, <vscale x 8 x i64> undef, <vscale x 8 x i32> zeroinitializer
  %ve = lshr <vscale x 8 x i64> %vd, %splat2
  %vf = trunc <vscale x 8 x i64> %ve to <vscale x 8 x i32>
  ret <vscale x 8 x i32> %vf
}

define <vscale x 8 x i32> @vmulhu_vi_nxv8i32_1(<vscale x 8 x i32> %va) {
; RV32-LABEL: vmulhu_vi_nxv8i32_1:
; RV32:       # %bb.0:
; RV32-NEXT:    vsetvli a0, zero, e32, m4, ta, mu
; RV32-NEXT:    vsrl.vi v8, v8, 28
; RV32-NEXT:    ret
;
; RV64-LABEL: vmulhu_vi_nxv8i32_1:
; RV64:       # %bb.0:
; RV64-NEXT:    li a0, 16
; RV64-NEXT:    vsetvli a1, zero, e32, m4, ta, mu
; RV64-NEXT:    vmulhu.vx v8, v8, a0
; RV64-NEXT:    ret
  %head1 = insertelement <vscale x 8 x i32> undef, i32 16, i32 0
  %splat1 = shufflevector <vscale x 8 x i32> %head1, <vscale x 8 x i32> undef, <vscale x 8 x i32> zeroinitializer
  %vb = zext <vscale x 8 x i32> %splat1 to <vscale x 8 x i64>
  %vc = zext <vscale x 8 x i32> %va to <vscale x 8 x i64>
  %vd = mul <vscale x 8 x i64> %vb, %vc
  %head2 = insertelement <vscale x 8 x i64> undef, i64 32, i32 0
  %splat2 = shufflevector <vscale x 8 x i64> %head2, <vscale x 8 x i64> undef, <vscale x 8 x i32> zeroinitializer
  %ve = lshr <vscale x 8 x i64> %vd, %splat2
  %vf = trunc <vscale x 8 x i64> %ve to <vscale x 8 x i32>
  ret <vscale x 8 x i32> %vf
}
