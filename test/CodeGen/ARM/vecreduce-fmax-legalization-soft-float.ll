; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=arm-none-eabi -mattr=-neon | FileCheck %s --check-prefix=CHECK

declare half @llvm.vector.reduce.fmax.v4f16(<4 x half>)
declare float @llvm.vector.reduce.fmax.v4f32(<4 x float>)
declare double @llvm.vector.reduce.fmax.v2f64(<2 x double>)
declare fp128 @llvm.vector.reduce.fmax.v2f128(<2 x fp128>)

define half @test_v4f16(<4 x half> %a) nounwind {
; CHECK-LABEL: test_v4f16:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    .save {r4, r5, r6, r7, r8, lr}
; CHECK-NEXT:    push {r4, r5, r6, r7, r8, lr}
; CHECK-NEXT:    mov r4, #255
; CHECK-NEXT:    mov r7, r0
; CHECK-NEXT:    orr r4, r4, #65280
; CHECK-NEXT:    mov r5, r2
; CHECK-NEXT:    and r0, r3, r4
; CHECK-NEXT:    mov r6, r1
; CHECK-NEXT:    bl __aeabi_h2f
; CHECK-NEXT:    mov r8, r0
; CHECK-NEXT:    and r0, r5, r4
; CHECK-NEXT:    bl __aeabi_h2f
; CHECK-NEXT:    mov r5, r0
; CHECK-NEXT:    and r0, r7, r4
; CHECK-NEXT:    bl __aeabi_h2f
; CHECK-NEXT:    mov r7, r0
; CHECK-NEXT:    and r0, r6, r4
; CHECK-NEXT:    bl __aeabi_h2f
; CHECK-NEXT:    mov r1, r0
; CHECK-NEXT:    mov r0, r7
; CHECK-NEXT:    bl fmaxf
; CHECK-NEXT:    mov r1, r5
; CHECK-NEXT:    bl fmaxf
; CHECK-NEXT:    mov r1, r8
; CHECK-NEXT:    bl fmaxf
; CHECK-NEXT:    bl __aeabi_f2h
; CHECK-NEXT:    pop {r4, r5, r6, r7, r8, lr}
; CHECK-NEXT:    mov pc, lr
  %b = call fast half @llvm.vector.reduce.fmax.v4f16(<4 x half> %a)
  ret half %b
}

define float @test_v4f32(<4 x float> %a) nounwind {
; CHECK-LABEL: test_v4f32:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    .save {r4, r5, r11, lr}
; CHECK-NEXT:    push {r4, r5, r11, lr}
; CHECK-NEXT:    mov r4, r3
; CHECK-NEXT:    mov r5, r2
; CHECK-NEXT:    bl fmaxf
; CHECK-NEXT:    mov r1, r5
; CHECK-NEXT:    bl fmaxf
; CHECK-NEXT:    mov r1, r4
; CHECK-NEXT:    bl fmaxf
; CHECK-NEXT:    pop {r4, r5, r11, lr}
; CHECK-NEXT:    mov pc, lr
  %b = call fast float @llvm.vector.reduce.fmax.v4f32(<4 x float> %a)
  ret float %b
}

define double @test_v2f64(<2 x double> %a) nounwind {
; CHECK-LABEL: test_v2f64:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    .save {r11, lr}
; CHECK-NEXT:    push {r11, lr}
; CHECK-NEXT:    bl fmax
; CHECK-NEXT:    pop {r11, lr}
; CHECK-NEXT:    mov pc, lr
  %b = call fast double @llvm.vector.reduce.fmax.v2f64(<2 x double> %a)
  ret double %b
}

define fp128 @test_v2f128(<2 x fp128> %a) nounwind {
; CHECK-LABEL: test_v2f128:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    .save {r11, lr}
; CHECK-NEXT:    push {r11, lr}
; CHECK-NEXT:    .pad #16
; CHECK-NEXT:    sub sp, sp, #16
; CHECK-NEXT:    ldr r12, [sp, #36]
; CHECK-NEXT:    str r12, [sp, #12]
; CHECK-NEXT:    ldr r12, [sp, #32]
; CHECK-NEXT:    str r12, [sp, #8]
; CHECK-NEXT:    ldr r12, [sp, #28]
; CHECK-NEXT:    str r12, [sp, #4]
; CHECK-NEXT:    ldr r12, [sp, #24]
; CHECK-NEXT:    str r12, [sp]
; CHECK-NEXT:    bl fmaxl
; CHECK-NEXT:    add sp, sp, #16
; CHECK-NEXT:    pop {r11, lr}
; CHECK-NEXT:    mov pc, lr
  %b = call fast fp128 @llvm.vector.reduce.fmax.v2f128(<2 x fp128> %a)
  ret fp128 %b
}
