; NOTE: Assertions have been autogenerated by utils/update_mir_test_checks.py
; RUN: llc -mtriple=x86_64-linux-gnu -O0 -global-isel -stop-after=irtranslator -verify-machineinstrs %s -o - 2>&1 | FileCheck %s

define i8 @zext_i1_to_i8(i1 %val) {
  ; CHECK-LABEL: name: zext_i1_to_i8
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY]](s32)
  ; CHECK:   [[ZEXT:%[0-9]+]]:_(s8) = G_ZEXT [[TRUNC]](s1)
  ; CHECK:   $al = COPY [[ZEXT]](s8)
  ; CHECK:   RET 0, implicit $al
  %res = zext i1 %val to i8
  ret i8 %res
}

define i16 @zext_i1_to_i16(i1 %val) {
  ; CHECK-LABEL: name: zext_i1_to_i16
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY]](s32)
  ; CHECK:   [[ZEXT:%[0-9]+]]:_(s16) = G_ZEXT [[TRUNC]](s1)
  ; CHECK:   $ax = COPY [[ZEXT]](s16)
  ; CHECK:   RET 0, implicit $ax
  %res = zext i1 %val to i16
  ret i16 %res
}

define i32 @zext_i1_to_i32(i1 %val) {
  ; CHECK-LABEL: name: zext_i1_to_i32
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY]](s32)
  ; CHECK:   [[ZEXT:%[0-9]+]]:_(s32) = G_ZEXT [[TRUNC]](s1)
  ; CHECK:   $eax = COPY [[ZEXT]](s32)
  ; CHECK:   RET 0, implicit $eax
  %res = zext i1 %val to i32
  ret i32 %res
}

define i64 @zext_i1_to_i64(i1 %val) {
  ; CHECK-LABEL: name: zext_i1_to_i64
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY]](s32)
  ; CHECK:   [[ZEXT:%[0-9]+]]:_(s64) = G_ZEXT [[TRUNC]](s1)
  ; CHECK:   $rax = COPY [[ZEXT]](s64)
  ; CHECK:   RET 0, implicit $rax
  %res = zext i1 %val to i64
  ret i64 %res
}

define i16 @zext_i8_to_i16(i8 %val) {
  ; CHECK-LABEL: name: zext_i8_to_i16
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[TRUNC:%[0-9]+]]:_(s8) = G_TRUNC [[COPY]](s32)
  ; CHECK:   [[ZEXT:%[0-9]+]]:_(s16) = G_ZEXT [[TRUNC]](s8)
  ; CHECK:   $ax = COPY [[ZEXT]](s16)
  ; CHECK:   RET 0, implicit $ax
  %res = zext i8 %val to i16
  ret i16 %res
}

define i32 @zext_i8_to_i32(i8 %val) {
  ; CHECK-LABEL: name: zext_i8_to_i32
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[TRUNC:%[0-9]+]]:_(s8) = G_TRUNC [[COPY]](s32)
  ; CHECK:   [[ZEXT:%[0-9]+]]:_(s32) = G_ZEXT [[TRUNC]](s8)
  ; CHECK:   $eax = COPY [[ZEXT]](s32)
  ; CHECK:   RET 0, implicit $eax
  %res = zext i8 %val to i32
  ret i32 %res
}

define i64 @zext_i8_to_i64(i8 %val) {
  ; CHECK-LABEL: name: zext_i8_to_i64
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[TRUNC:%[0-9]+]]:_(s8) = G_TRUNC [[COPY]](s32)
  ; CHECK:   [[ZEXT:%[0-9]+]]:_(s64) = G_ZEXT [[TRUNC]](s8)
  ; CHECK:   $rax = COPY [[ZEXT]](s64)
  ; CHECK:   RET 0, implicit $rax
  %res = zext i8 %val to i64
  ret i64 %res
}

define i32 @zext_i16_to_i32(i16 %val) {
  ; CHECK-LABEL: name: zext_i16_to_i32
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[TRUNC:%[0-9]+]]:_(s16) = G_TRUNC [[COPY]](s32)
  ; CHECK:   [[ZEXT:%[0-9]+]]:_(s32) = G_ZEXT [[TRUNC]](s16)
  ; CHECK:   $eax = COPY [[ZEXT]](s32)
  ; CHECK:   RET 0, implicit $eax
  %res = zext i16 %val to i32
  ret i32 %res
}

define i64 @zext_i16_to_i64(i16 %val) {
  ; CHECK-LABEL: name: zext_i16_to_i64
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[TRUNC:%[0-9]+]]:_(s16) = G_TRUNC [[COPY]](s32)
  ; CHECK:   [[ZEXT:%[0-9]+]]:_(s64) = G_ZEXT [[TRUNC]](s16)
  ; CHECK:   $rax = COPY [[ZEXT]](s64)
  ; CHECK:   RET 0, implicit $rax
  %res = zext i16 %val to i64
  ret i64 %res
}

define i64 @zext_i32_to_i64(i32 %val) {
  ; CHECK-LABEL: name: zext_i32_to_i64
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[ZEXT:%[0-9]+]]:_(s64) = G_ZEXT [[COPY]](s32)
  ; CHECK:   $rax = COPY [[ZEXT]](s64)
  ; CHECK:   RET 0, implicit $rax
  %res = zext i32 %val to i64
  ret i64 %res
}

define i8 @test_sdiv_i8(i8 %arg1, i8 %arg2) {
  ; CHECK-LABEL: name: test_sdiv_i8
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi, $esi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[TRUNC:%[0-9]+]]:_(s8) = G_TRUNC [[COPY]](s32)
  ; CHECK:   [[COPY1:%[0-9]+]]:_(s32) = COPY $esi
  ; CHECK:   [[TRUNC1:%[0-9]+]]:_(s8) = G_TRUNC [[COPY1]](s32)
  ; CHECK:   [[SDIV:%[0-9]+]]:_(s8) = G_SDIV [[TRUNC]], [[TRUNC1]]
  ; CHECK:   $al = COPY [[SDIV]](s8)
  ; CHECK:   RET 0, implicit $al
  %res = sdiv i8 %arg1, %arg2
  ret i8 %res
}

define i16 @test_sdiv_i16(i16 %arg1, i16 %arg2) {
  ; CHECK-LABEL: name: test_sdiv_i16
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi, $esi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[TRUNC:%[0-9]+]]:_(s16) = G_TRUNC [[COPY]](s32)
  ; CHECK:   [[COPY1:%[0-9]+]]:_(s32) = COPY $esi
  ; CHECK:   [[TRUNC1:%[0-9]+]]:_(s16) = G_TRUNC [[COPY1]](s32)
  ; CHECK:   [[SDIV:%[0-9]+]]:_(s16) = G_SDIV [[TRUNC]], [[TRUNC1]]
  ; CHECK:   $ax = COPY [[SDIV]](s16)
  ; CHECK:   RET 0, implicit $ax
  %res = sdiv i16 %arg1, %arg2
  ret i16 %res
}

define i32 @test_sdiv_i32(i32 %arg1, i32 %arg2) {
  ; CHECK-LABEL: name: test_sdiv_i32
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi, $esi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[COPY1:%[0-9]+]]:_(s32) = COPY $esi
  ; CHECK:   [[SDIV:%[0-9]+]]:_(s32) = G_SDIV [[COPY]], [[COPY1]]
  ; CHECK:   $eax = COPY [[SDIV]](s32)
  ; CHECK:   RET 0, implicit $eax
  %res = sdiv i32 %arg1, %arg2
  ret i32 %res
}

define i64 @test_sdiv_i64(i64 %arg1, i64 %arg2) {
  ; CHECK-LABEL: name: test_sdiv_i64
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $rdi, $rsi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s64) = COPY $rdi
  ; CHECK:   [[COPY1:%[0-9]+]]:_(s64) = COPY $rsi
  ; CHECK:   [[SDIV:%[0-9]+]]:_(s64) = G_SDIV [[COPY]], [[COPY1]]
  ; CHECK:   $rax = COPY [[SDIV]](s64)
  ; CHECK:   RET 0, implicit $rax
  %res = sdiv i64 %arg1, %arg2
  ret i64 %res
}
define float @test_fptrunc(double %in) {
  ; CHECK-LABEL: name: test_fptrunc
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $xmm0
  ; CHECK:   [[COPY:%[0-9]+]]:_(s64) = COPY $xmm0
  ; CHECK:   [[FPTRUNC:%[0-9]+]]:_(s32) = G_FPTRUNC [[COPY]](s64)
  ; CHECK:   $xmm0 = COPY [[FPTRUNC]](s32)
  ; CHECK:   RET 0, implicit $xmm0
  %res = fptrunc double %in to float
  ret float %res
}

define i8 @test_srem_i8(i8 %arg1, i8 %arg2) {
  ; CHECK-LABEL: name: test_srem_i8
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi, $esi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[TRUNC:%[0-9]+]]:_(s8) = G_TRUNC [[COPY]](s32)
  ; CHECK:   [[COPY1:%[0-9]+]]:_(s32) = COPY $esi
  ; CHECK:   [[TRUNC1:%[0-9]+]]:_(s8) = G_TRUNC [[COPY1]](s32)
  ; CHECK:   [[SREM:%[0-9]+]]:_(s8) = G_SREM [[TRUNC]], [[TRUNC1]]
  ; CHECK:   $al = COPY [[SREM]](s8)
  ; CHECK:   RET 0, implicit $al
  %res = srem i8 %arg1, %arg2
  ret i8 %res
}

define i16 @test_srem_i16(i16 %arg1, i16 %arg2) {
  ; CHECK-LABEL: name: test_srem_i16
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi, $esi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[TRUNC:%[0-9]+]]:_(s16) = G_TRUNC [[COPY]](s32)
  ; CHECK:   [[COPY1:%[0-9]+]]:_(s32) = COPY $esi
  ; CHECK:   [[TRUNC1:%[0-9]+]]:_(s16) = G_TRUNC [[COPY1]](s32)
  ; CHECK:   [[SREM:%[0-9]+]]:_(s16) = G_SREM [[TRUNC]], [[TRUNC1]]
  ; CHECK:   $ax = COPY [[SREM]](s16)
  ; CHECK:   RET 0, implicit $ax
  %res = srem i16 %arg1, %arg2
  ret i16 %res
}

define i32 @test_srem_i32(i32 %arg1, i32 %arg2) {
  ; CHECK-LABEL: name: test_srem_i32
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi, $esi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[COPY1:%[0-9]+]]:_(s32) = COPY $esi
  ; CHECK:   [[SREM:%[0-9]+]]:_(s32) = G_SREM [[COPY]], [[COPY1]]
  ; CHECK:   $eax = COPY [[SREM]](s32)
  ; CHECK:   RET 0, implicit $eax
  %res = srem i32 %arg1, %arg2
  ret i32 %res
}

define i64 @test_srem_i64(i64 %arg1, i64 %arg2) {
  ; CHECK-LABEL: name: test_srem_i64
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $rdi, $rsi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s64) = COPY $rdi
  ; CHECK:   [[COPY1:%[0-9]+]]:_(s64) = COPY $rsi
  ; CHECK:   [[SREM:%[0-9]+]]:_(s64) = G_SREM [[COPY]], [[COPY1]]
  ; CHECK:   $rax = COPY [[SREM]](s64)
  ; CHECK:   RET 0, implicit $rax
  %res = srem i64 %arg1, %arg2
  ret i64 %res
}

define i8 @test_udiv_i8(i8 %arg1, i8 %arg2) {
  ; CHECK-LABEL: name: test_udiv_i8
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi, $esi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[TRUNC:%[0-9]+]]:_(s8) = G_TRUNC [[COPY]](s32)
  ; CHECK:   [[COPY1:%[0-9]+]]:_(s32) = COPY $esi
  ; CHECK:   [[TRUNC1:%[0-9]+]]:_(s8) = G_TRUNC [[COPY1]](s32)
  ; CHECK:   [[UDIV:%[0-9]+]]:_(s8) = G_UDIV [[TRUNC]], [[TRUNC1]]
  ; CHECK:   $al = COPY [[UDIV]](s8)
  ; CHECK:   RET 0, implicit $al
  %res = udiv i8 %arg1, %arg2
  ret i8 %res
}

define i16 @test_udiv_i16(i16 %arg1, i16 %arg2) {
  ; CHECK-LABEL: name: test_udiv_i16
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi, $esi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[TRUNC:%[0-9]+]]:_(s16) = G_TRUNC [[COPY]](s32)
  ; CHECK:   [[COPY1:%[0-9]+]]:_(s32) = COPY $esi
  ; CHECK:   [[TRUNC1:%[0-9]+]]:_(s16) = G_TRUNC [[COPY1]](s32)
  ; CHECK:   [[UDIV:%[0-9]+]]:_(s16) = G_UDIV [[TRUNC]], [[TRUNC1]]
  ; CHECK:   $ax = COPY [[UDIV]](s16)
  ; CHECK:   RET 0, implicit $ax
  %res = udiv i16 %arg1, %arg2
  ret i16 %res
}

define i32 @test_udiv_i32(i32 %arg1, i32 %arg2) {
  ; CHECK-LABEL: name: test_udiv_i32
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi, $esi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[COPY1:%[0-9]+]]:_(s32) = COPY $esi
  ; CHECK:   [[UDIV:%[0-9]+]]:_(s32) = G_UDIV [[COPY]], [[COPY1]]
  ; CHECK:   $eax = COPY [[UDIV]](s32)
  ; CHECK:   RET 0, implicit $eax
  %res = udiv i32 %arg1, %arg2
  ret i32 %res
}

define i64 @test_udiv_i64(i64 %arg1, i64 %arg2) {
  ; CHECK-LABEL: name: test_udiv_i64
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $rdi, $rsi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s64) = COPY $rdi
  ; CHECK:   [[COPY1:%[0-9]+]]:_(s64) = COPY $rsi
  ; CHECK:   [[UDIV:%[0-9]+]]:_(s64) = G_UDIV [[COPY]], [[COPY1]]
  ; CHECK:   $rax = COPY [[UDIV]](s64)
  ; CHECK:   RET 0, implicit $rax
  %res = udiv i64 %arg1, %arg2
  ret i64 %res
}

define i8 @test_urem_i8(i8 %arg1, i8 %arg2) {
  ; CHECK-LABEL: name: test_urem_i8
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi, $esi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[TRUNC:%[0-9]+]]:_(s8) = G_TRUNC [[COPY]](s32)
  ; CHECK:   [[COPY1:%[0-9]+]]:_(s32) = COPY $esi
  ; CHECK:   [[TRUNC1:%[0-9]+]]:_(s8) = G_TRUNC [[COPY1]](s32)
  ; CHECK:   [[UREM:%[0-9]+]]:_(s8) = G_UREM [[TRUNC]], [[TRUNC1]]
  ; CHECK:   $al = COPY [[UREM]](s8)
  ; CHECK:   RET 0, implicit $al
  %res = urem i8 %arg1, %arg2
  ret i8 %res
}

define i16 @test_urem_i16(i16 %arg1, i16 %arg2) {
  ; CHECK-LABEL: name: test_urem_i16
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi, $esi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[TRUNC:%[0-9]+]]:_(s16) = G_TRUNC [[COPY]](s32)
  ; CHECK:   [[COPY1:%[0-9]+]]:_(s32) = COPY $esi
  ; CHECK:   [[TRUNC1:%[0-9]+]]:_(s16) = G_TRUNC [[COPY1]](s32)
  ; CHECK:   [[UREM:%[0-9]+]]:_(s16) = G_UREM [[TRUNC]], [[TRUNC1]]
  ; CHECK:   $ax = COPY [[UREM]](s16)
  ; CHECK:   RET 0, implicit $ax
  %res = urem i16 %arg1, %arg2
  ret i16 %res
}

define i32 @test_urem_i32(i32 %arg1, i32 %arg2) {
  ; CHECK-LABEL: name: test_urem_i32
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $edi, $esi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $edi
  ; CHECK:   [[COPY1:%[0-9]+]]:_(s32) = COPY $esi
  ; CHECK:   [[UREM:%[0-9]+]]:_(s32) = G_UREM [[COPY]], [[COPY1]]
  ; CHECK:   $eax = COPY [[UREM]](s32)
  ; CHECK:   RET 0, implicit $eax
  %res = urem i32 %arg1, %arg2
  ret i32 %res
}

define i64 @test_urem_i64(i64 %arg1, i64 %arg2) {
  ; CHECK-LABEL: name: test_urem_i64
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   liveins: $rdi, $rsi
  ; CHECK:   [[COPY:%[0-9]+]]:_(s64) = COPY $rdi
  ; CHECK:   [[COPY1:%[0-9]+]]:_(s64) = COPY $rsi
  ; CHECK:   [[UREM:%[0-9]+]]:_(s64) = G_UREM [[COPY]], [[COPY1]]
  ; CHECK:   $rax = COPY [[UREM]](s64)
  ; CHECK:   RET 0, implicit $rax
  %res = urem i64 %arg1, %arg2
  ret i64 %res
}
