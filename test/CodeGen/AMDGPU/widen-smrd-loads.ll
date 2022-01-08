; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -amdgpu-codegenprepare-widen-constant-loads=0 -mtriple=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefix=SI %s
; RUN: llc -amdgpu-codegenprepare-widen-constant-loads=0 -mtriple=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefix=VI %s

define amdgpu_kernel void @widen_i16_constant_load(i16 addrspace(4)* %arg) {
; SI-LABEL: widen_i16_constant_load:
; SI:       ; %bb.0:
; SI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x9
; SI-NEXT:    s_mov_b32 s3, 0xf000
; SI-NEXT:    s_mov_b32 s2, -1
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_load_dword s1, s[0:1], 0x0
; SI-NEXT:    s_mov_b32 s0, 0
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_addk_i32 s1, 0x3e7
; SI-NEXT:    s_or_b32 s4, s1, 4
; SI-NEXT:    s_mov_b32 s1, s0
; SI-NEXT:    v_mov_b32_e32 v0, s4
; SI-NEXT:    buffer_store_short v0, off, s[0:3], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: widen_i16_constant_load:
; VI:       ; %bb.0:
; VI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x24
; VI-NEXT:    v_mov_b32_e32 v0, 0
; VI-NEXT:    v_mov_b32_e32 v1, 0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_load_dword s0, s[0:1], 0x0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_addk_i32 s0, 0x3e7
; VI-NEXT:    s_or_b32 s0, s0, 4
; VI-NEXT:    v_mov_b32_e32 v2, s0
; VI-NEXT:    flat_store_short v[0:1], v2
; VI-NEXT:    s_endpgm
  %load = load i16, i16 addrspace(4)* %arg, align 4
  %add = add i16 %load, 999
  %or = or i16 %add, 4
  store i16 %or, i16 addrspace(1)* null
  ret void
}

define amdgpu_kernel void @widen_i16_constant_load_zext_i32(i16 addrspace(4)* %arg) {
; SI-LABEL: widen_i16_constant_load_zext_i32:
; SI:       ; %bb.0:
; SI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x9
; SI-NEXT:    s_mov_b32 s3, 0xf000
; SI-NEXT:    s_mov_b32 s2, -1
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_load_dword s1, s[0:1], 0x0
; SI-NEXT:    s_mov_b32 s0, 0
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_and_b32 s1, s1, 0xffff
; SI-NEXT:    s_addk_i32 s1, 0x3e7
; SI-NEXT:    s_or_b32 s4, s1, 4
; SI-NEXT:    s_mov_b32 s1, s0
; SI-NEXT:    v_mov_b32_e32 v0, s4
; SI-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: widen_i16_constant_load_zext_i32:
; VI:       ; %bb.0:
; VI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x24
; VI-NEXT:    v_mov_b32_e32 v0, 0
; VI-NEXT:    v_mov_b32_e32 v1, 0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_load_dword s0, s[0:1], 0x0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_and_b32 s0, s0, 0xffff
; VI-NEXT:    s_addk_i32 s0, 0x3e7
; VI-NEXT:    s_or_b32 s0, s0, 4
; VI-NEXT:    v_mov_b32_e32 v2, s0
; VI-NEXT:    flat_store_dword v[0:1], v2
; VI-NEXT:    s_endpgm
  %load = load i16, i16 addrspace(4)* %arg, align 4
  %ext = zext i16 %load to i32
  %add = add i32 %ext, 999
  %or = or i32 %add, 4
  store i32 %or, i32 addrspace(1)* null
  ret void
}

define amdgpu_kernel void @widen_i16_constant_load_sext_i32(i16 addrspace(4)* %arg) {
; SI-LABEL: widen_i16_constant_load_sext_i32:
; SI:       ; %bb.0:
; SI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x9
; SI-NEXT:    s_mov_b32 s3, 0xf000
; SI-NEXT:    s_mov_b32 s2, -1
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_load_dword s1, s[0:1], 0x0
; SI-NEXT:    s_mov_b32 s0, 0
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_sext_i32_i16 s1, s1
; SI-NEXT:    s_addk_i32 s1, 0x3e7
; SI-NEXT:    s_or_b32 s4, s1, 4
; SI-NEXT:    s_mov_b32 s1, s0
; SI-NEXT:    v_mov_b32_e32 v0, s4
; SI-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: widen_i16_constant_load_sext_i32:
; VI:       ; %bb.0:
; VI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x24
; VI-NEXT:    v_mov_b32_e32 v0, 0
; VI-NEXT:    v_mov_b32_e32 v1, 0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_load_dword s0, s[0:1], 0x0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_sext_i32_i16 s0, s0
; VI-NEXT:    s_addk_i32 s0, 0x3e7
; VI-NEXT:    s_or_b32 s0, s0, 4
; VI-NEXT:    v_mov_b32_e32 v2, s0
; VI-NEXT:    flat_store_dword v[0:1], v2
; VI-NEXT:    s_endpgm
  %load = load i16, i16 addrspace(4)* %arg, align 4
  %ext = sext i16 %load to i32
  %add = add i32 %ext, 999
  %or = or i32 %add, 4
  store i32 %or, i32 addrspace(1)* null
  ret void
}

define amdgpu_kernel void @widen_i17_constant_load(i17 addrspace(4)* %arg) {
; SI-LABEL: widen_i17_constant_load:
; SI:       ; %bb.0:
; SI-NEXT:    s_load_dwordx2 s[4:5], s[0:1], 0x9
; SI-NEXT:    s_mov_b32 s0, 0
; SI-NEXT:    s_mov_b32 s3, 0xf000
; SI-NEXT:    s_mov_b32 s2, -1
; SI-NEXT:    s_mov_b32 s1, s0
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_load_dword s7, s[4:5], 0x0
; SI-NEXT:    s_mov_b32 s4, 2
; SI-NEXT:    s_mov_b32 s5, s0
; SI-NEXT:    s_mov_b32 s6, s2
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_add_i32 s7, s7, 34
; SI-NEXT:    s_or_b32 s7, s7, 4
; SI-NEXT:    v_mov_b32_e32 v0, s7
; SI-NEXT:    s_bfe_u32 s8, s7, 0x10010
; SI-NEXT:    buffer_store_short v0, off, s[0:3], 0
; SI-NEXT:    s_mov_b32 s7, s3
; SI-NEXT:    s_waitcnt expcnt(0)
; SI-NEXT:    v_mov_b32_e32 v0, s8
; SI-NEXT:    buffer_store_byte v0, off, s[4:7], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: widen_i17_constant_load:
; VI:       ; %bb.0:
; VI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x24
; VI-NEXT:    v_mov_b32_e32 v0, 0
; VI-NEXT:    v_mov_b32_e32 v1, 0
; VI-NEXT:    v_mov_b32_e32 v2, 2
; VI-NEXT:    v_mov_b32_e32 v3, 0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_load_dword s0, s[0:1], 0x0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_add_i32 s0, s0, 34
; VI-NEXT:    s_or_b32 s0, s0, 4
; VI-NEXT:    v_mov_b32_e32 v4, s0
; VI-NEXT:    s_bfe_u32 s0, s0, 0x10010
; VI-NEXT:    flat_store_short v[0:1], v4
; VI-NEXT:    v_mov_b32_e32 v0, s0
; VI-NEXT:    flat_store_byte v[2:3], v0
; VI-NEXT:    s_endpgm
  %load = load i17, i17 addrspace(4)* %arg, align 4
  %add = add i17 %load, 34
  %or = or i17 %add, 4
  store i17 %or, i17 addrspace(1)* null
  ret void
}

define amdgpu_kernel void @widen_f16_constant_load(half addrspace(4)* %arg) {
; SI-LABEL: widen_f16_constant_load:
; SI:       ; %bb.0:
; SI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x9
; SI-NEXT:    s_mov_b32 s3, 0xf000
; SI-NEXT:    s_mov_b32 s2, -1
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_load_dword s0, s[0:1], 0x0
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    v_cvt_f32_f16_e32 v0, s0
; SI-NEXT:    s_mov_b32 s0, 0
; SI-NEXT:    s_mov_b32 s1, s0
; SI-NEXT:    v_add_f32_e32 v0, 4.0, v0
; SI-NEXT:    v_cvt_f16_f32_e32 v0, v0
; SI-NEXT:    buffer_store_short v0, off, s[0:3], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: widen_f16_constant_load:
; VI:       ; %bb.0:
; VI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x24
; VI-NEXT:    v_mov_b32_e32 v0, 0
; VI-NEXT:    v_mov_b32_e32 v1, 0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_load_dword s0, s[0:1], 0x0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    v_add_f16_e64 v2, s0, 4.0
; VI-NEXT:    flat_store_short v[0:1], v2
; VI-NEXT:    s_endpgm
  %load = load half, half addrspace(4)* %arg, align 4
  %add = fadd half %load, 4.0
  store half %add, half addrspace(1)* null
  ret void
}

; FIXME: valu usage on VI
define amdgpu_kernel void @widen_v2i8_constant_load(<2 x i8> addrspace(4)* %arg) {
; SI-LABEL: widen_v2i8_constant_load:
; SI:       ; %bb.0:
; SI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x9
; SI-NEXT:    s_mov_b32 s3, 0xf000
; SI-NEXT:    s_mov_b32 s2, -1
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_load_dword s1, s[0:1], 0x0
; SI-NEXT:    s_mov_b32 s0, 0
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_and_b32 s4, s1, 0xff00
; SI-NEXT:    s_add_i32 s1, s1, 12
; SI-NEXT:    s_or_b32 s1, s1, 4
; SI-NEXT:    s_and_b32 s1, s1, 0xff
; SI-NEXT:    s_or_b32 s1, s4, s1
; SI-NEXT:    s_addk_i32 s1, 0x2c00
; SI-NEXT:    s_or_b32 s4, s1, 0x300
; SI-NEXT:    s_mov_b32 s1, s0
; SI-NEXT:    v_mov_b32_e32 v0, s4
; SI-NEXT:    buffer_store_short v0, off, s[0:3], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: widen_v2i8_constant_load:
; VI:       ; %bb.0:
; VI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x24
; VI-NEXT:    v_mov_b32_e32 v0, 44
; VI-NEXT:    v_mov_b32_e32 v1, 3
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_load_dword s0, s[0:1], 0x0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_and_b32 s1, s0, 0xffff
; VI-NEXT:    v_mov_b32_e32 v2, s0
; VI-NEXT:    s_add_i32 s1, s1, 12
; VI-NEXT:    v_add_u32_sdwa v0, vcc, v0, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1
; VI-NEXT:    s_or_b32 s0, s1, 4
; VI-NEXT:    v_or_b32_sdwa v2, v0, v1 dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-NEXT:    v_mov_b32_e32 v3, s0
; VI-NEXT:    v_mov_b32_e32 v0, 0
; VI-NEXT:    v_mov_b32_e32 v1, 0
; VI-NEXT:    v_or_b32_sdwa v2, v3, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; VI-NEXT:    flat_store_short v[0:1], v2
; VI-NEXT:    s_endpgm
  %load = load <2 x i8>, <2 x i8> addrspace(4)* %arg, align 4
  %add = add <2 x i8> %load, <i8 12, i8 44>
  %or = or <2 x i8> %add, <i8 4, i8 3>
  store <2 x i8> %or, <2 x i8> addrspace(1)* null
  ret void
}

define amdgpu_kernel void @no_widen_i16_constant_divergent_load(i16 addrspace(4)* %arg) {
; SI-LABEL: no_widen_i16_constant_divergent_load:
; SI:       ; %bb.0:
; SI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x9
; SI-NEXT:    s_mov_b32 s2, 0
; SI-NEXT:    s_mov_b32 s3, 0xf000
; SI-NEXT:    v_lshlrev_b32_e32 v0, 1, v0
; SI-NEXT:    v_mov_b32_e32 v1, 0
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    buffer_load_ushort v0, v[0:1], s[0:3], 0 addr64
; SI-NEXT:    s_mov_b32 s6, -1
; SI-NEXT:    s_mov_b32 s4, s2
; SI-NEXT:    s_mov_b32 s5, s2
; SI-NEXT:    s_mov_b32 s7, s3
; SI-NEXT:    s_waitcnt vmcnt(0)
; SI-NEXT:    v_add_i32_e32 v0, vcc, 0x3e7, v0
; SI-NEXT:    v_or_b32_e32 v0, 4, v0
; SI-NEXT:    buffer_store_short v0, off, s[4:7], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: no_widen_i16_constant_divergent_load:
; VI:       ; %bb.0:
; VI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x24
; VI-NEXT:    v_lshlrev_b32_e32 v0, 1, v0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    v_mov_b32_e32 v1, s1
; VI-NEXT:    v_add_u32_e32 v0, vcc, s0, v0
; VI-NEXT:    v_addc_u32_e32 v1, vcc, 0, v1, vcc
; VI-NEXT:    flat_load_ushort v0, v[0:1]
; VI-NEXT:    s_waitcnt vmcnt(0)
; VI-NEXT:    v_add_u16_e32 v2, 0x3e7, v0
; VI-NEXT:    v_mov_b32_e32 v0, 0
; VI-NEXT:    v_mov_b32_e32 v1, 0
; VI-NEXT:    v_or_b32_e32 v2, 4, v2
; VI-NEXT:    flat_store_short v[0:1], v2
; VI-NEXT:    s_endpgm
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = zext i32 %tid to i64
  %gep.arg = getelementptr inbounds i16, i16 addrspace(4)* %arg, i64 %tid.ext
  %load = load i16, i16 addrspace(4)* %gep.arg, align 4
  %add = add i16 %load, 999
  %or = or i16 %add, 4
  store i16 %or, i16 addrspace(1)* null
  ret void
}

define amdgpu_kernel void @widen_i1_constant_load(i1 addrspace(4)* %arg) {
; SI-LABEL: widen_i1_constant_load:
; SI:       ; %bb.0:
; SI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x9
; SI-NEXT:    s_mov_b32 s3, 0xf000
; SI-NEXT:    s_mov_b32 s2, -1
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_load_dword s1, s[0:1], 0x0
; SI-NEXT:    s_mov_b32 s0, 0
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_and_b32 s4, s1, 1
; SI-NEXT:    s_mov_b32 s1, s0
; SI-NEXT:    v_mov_b32_e32 v0, s4
; SI-NEXT:    buffer_store_byte v0, off, s[0:3], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: widen_i1_constant_load:
; VI:       ; %bb.0:
; VI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x24
; VI-NEXT:    v_mov_b32_e32 v0, 0
; VI-NEXT:    v_mov_b32_e32 v1, 0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_load_dword s0, s[0:1], 0x0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_and_b32 s0, s0, 1
; VI-NEXT:    v_mov_b32_e32 v2, s0
; VI-NEXT:    flat_store_byte v[0:1], v2
; VI-NEXT:    s_endpgm
  %load = load i1, i1 addrspace(4)* %arg, align 4
  %and = and i1 %load, true
  store i1 %and, i1 addrspace(1)* null
  ret void
}

define amdgpu_kernel void @widen_i16_zextload_i64_constant_load(i16 addrspace(4)* %arg) {
; SI-LABEL: widen_i16_zextload_i64_constant_load:
; SI:       ; %bb.0:
; SI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x9
; SI-NEXT:    s_mov_b32 s3, 0xf000
; SI-NEXT:    s_mov_b32 s2, -1
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_load_dword s1, s[0:1], 0x0
; SI-NEXT:    s_mov_b32 s0, 0
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_and_b32 s1, s1, 0xffff
; SI-NEXT:    s_addk_i32 s1, 0x3e7
; SI-NEXT:    s_or_b32 s4, s1, 4
; SI-NEXT:    s_mov_b32 s1, s0
; SI-NEXT:    v_mov_b32_e32 v0, s4
; SI-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: widen_i16_zextload_i64_constant_load:
; VI:       ; %bb.0:
; VI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x24
; VI-NEXT:    v_mov_b32_e32 v0, 0
; VI-NEXT:    v_mov_b32_e32 v1, 0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_load_dword s0, s[0:1], 0x0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_and_b32 s0, s0, 0xffff
; VI-NEXT:    s_addk_i32 s0, 0x3e7
; VI-NEXT:    s_or_b32 s0, s0, 4
; VI-NEXT:    v_mov_b32_e32 v2, s0
; VI-NEXT:    flat_store_dword v[0:1], v2
; VI-NEXT:    s_endpgm
  %load = load i16, i16 addrspace(4)* %arg, align 4
  %zext = zext i16 %load to i32
  %add = add i32 %zext, 999
  %or = or i32 %add, 4
  store i32 %or, i32 addrspace(1)* null
  ret void
}

define amdgpu_kernel void @widen_i1_zext_to_i64_constant_load(i1 addrspace(4)* %arg) {
; SI-LABEL: widen_i1_zext_to_i64_constant_load:
; SI:       ; %bb.0:
; SI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x9
; SI-NEXT:    s_mov_b32 s3, 0xf000
; SI-NEXT:    s_mov_b32 s2, -1
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_load_dword s1, s[0:1], 0x0
; SI-NEXT:    s_mov_b32 s0, 0
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_and_b32 s1, s1, 1
; SI-NEXT:    s_add_u32 s4, s1, 0x3e7
; SI-NEXT:    s_addc_u32 s5, 0, 0
; SI-NEXT:    v_mov_b32_e32 v0, s4
; SI-NEXT:    s_mov_b32 s1, s0
; SI-NEXT:    v_mov_b32_e32 v1, s5
; SI-NEXT:    buffer_store_dwordx2 v[0:1], off, s[0:3], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: widen_i1_zext_to_i64_constant_load:
; VI:       ; %bb.0:
; VI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x24
; VI-NEXT:    v_mov_b32_e32 v0, 0
; VI-NEXT:    v_mov_b32_e32 v1, 0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_load_dword s0, s[0:1], 0x0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_and_b32 s0, s0, 1
; VI-NEXT:    s_add_u32 s0, s0, 0x3e7
; VI-NEXT:    s_addc_u32 s1, 0, 0
; VI-NEXT:    v_mov_b32_e32 v3, s1
; VI-NEXT:    v_mov_b32_e32 v2, s0
; VI-NEXT:    flat_store_dwordx2 v[0:1], v[2:3]
; VI-NEXT:    s_endpgm
  %load = load i1, i1 addrspace(4)* %arg, align 4
  %zext = zext i1 %load to i64
  %add = add i64 %zext, 999
  store i64 %add, i64 addrspace(1)* null
  ret void
}

define amdgpu_kernel void @widen_i16_constant32_load(i16 addrspace(6)* %arg) {
; SI-LABEL: widen_i16_constant32_load:
; SI:       ; %bb.0:
; SI-NEXT:    s_load_dword s0, s[0:1], 0x9
; SI-NEXT:    s_mov_b32 s1, 0
; SI-NEXT:    s_mov_b32 s3, 0xf000
; SI-NEXT:    s_mov_b32 s2, -1
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_load_dword s0, s[0:1], 0x0
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_addk_i32 s0, 0x3e7
; SI-NEXT:    s_or_b32 s4, s0, 4
; SI-NEXT:    s_mov_b32 s0, s1
; SI-NEXT:    v_mov_b32_e32 v0, s4
; SI-NEXT:    buffer_store_short v0, off, s[0:3], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: widen_i16_constant32_load:
; VI:       ; %bb.0:
; VI-NEXT:    s_load_dword s0, s[0:1], 0x24
; VI-NEXT:    s_mov_b32 s1, 0
; VI-NEXT:    v_mov_b32_e32 v0, 0
; VI-NEXT:    v_mov_b32_e32 v1, 0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_load_dword s0, s[0:1], 0x0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_addk_i32 s0, 0x3e7
; VI-NEXT:    s_or_b32 s0, s0, 4
; VI-NEXT:    v_mov_b32_e32 v2, s0
; VI-NEXT:    flat_store_short v[0:1], v2
; VI-NEXT:    s_endpgm
  %load = load i16, i16 addrspace(6)* %arg, align 4
  %add = add i16 %load, 999
  %or = or i16 %add, 4
  store i16 %or, i16 addrspace(1)* null
  ret void
}

define amdgpu_kernel void @widen_i16_global_invariant_load(i16 addrspace(1)* %arg) {
; SI-LABEL: widen_i16_global_invariant_load:
; SI:       ; %bb.0:
; SI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x9
; SI-NEXT:    s_mov_b32 s3, 0xf000
; SI-NEXT:    s_mov_b32 s2, -1
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_load_dword s1, s[0:1], 0x0
; SI-NEXT:    s_mov_b32 s0, 0
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_addk_i32 s1, 0x3e7
; SI-NEXT:    s_or_b32 s4, s1, 1
; SI-NEXT:    s_mov_b32 s1, s0
; SI-NEXT:    v_mov_b32_e32 v0, s4
; SI-NEXT:    buffer_store_short v0, off, s[0:3], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: widen_i16_global_invariant_load:
; VI:       ; %bb.0:
; VI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x24
; VI-NEXT:    v_mov_b32_e32 v0, 0
; VI-NEXT:    v_mov_b32_e32 v1, 0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_load_dword s0, s[0:1], 0x0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_addk_i32 s0, 0x3e7
; VI-NEXT:    s_or_b32 s0, s0, 1
; VI-NEXT:    v_mov_b32_e32 v2, s0
; VI-NEXT:    flat_store_short v[0:1], v2
; VI-NEXT:    s_endpgm
  %load = load i16, i16 addrspace(1)* %arg, align 4, !invariant.load !0
  %add = add i16 %load, 999
  %or = or i16 %add, 1
  store i16 %or, i16 addrspace(1)* null
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()

!0 = !{}
