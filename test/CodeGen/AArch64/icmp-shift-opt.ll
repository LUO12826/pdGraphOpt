; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s

; Optimize expanded SRL/SHL used as an input of
; SETCC comparing it with zero by removing rotation.
;
; See https://bugs.llvm.org/show_bug.cgi?id=50197
define i128 @opt_setcc_lt_power_of_2(i128 %a) nounwind {
; CHECK-LABEL: opt_setcc_lt_power_of_2:
; CHECK:       // %bb.0:
; CHECK-NEXT:  .LBB0_1: // %loop
; CHECK-NEXT:    // =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    adds x0, x0, #1
; CHECK-NEXT:    adcs x1, x1, xzr
; CHECK-NEXT:    extr x8, x1, x0, #60
; CHECK-NEXT:    orr x8, x8, x1, lsr #60
; CHECK-NEXT:    cbnz x8, .LBB0_1
; CHECK-NEXT:  // %bb.2: // %exit
; CHECK-NEXT:    ret
  br label %loop

loop:
  %phi.a = phi i128 [ %a, %0 ], [ %inc, %loop ]
  %inc = add i128 %phi.a, 1
  %cmp = icmp ult i128 %inc, 1152921504606846976
  br i1 %cmp, label %exit, label %loop

exit:
  ret i128 %inc
}

define i1 @opt_setcc_srl_eq_zero(i128 %a) nounwind {
; CHECK-LABEL: opt_setcc_srl_eq_zero:
; CHECK:       // %bb.0:
; CHECK-NEXT:    extr x8, x1, x0, #17
; CHECK-NEXT:    orr x8, x8, x1, lsr #17
; CHECK-NEXT:    cmp x8, #0
; CHECK-NEXT:    cset w0, eq
; CHECK-NEXT:    ret
   %srl = lshr i128 %a, 17
   %cmp = icmp eq i128 %srl, 0
   ret i1 %cmp
}

define i1 @opt_setcc_srl_ne_zero(i128 %a) nounwind {
; CHECK-LABEL: opt_setcc_srl_ne_zero:
; CHECK:       // %bb.0:
; CHECK-NEXT:    extr x8, x1, x0, #17
; CHECK-NEXT:    orr x8, x8, x1, lsr #17
; CHECK-NEXT:    cmp x8, #0
; CHECK-NEXT:    cset w0, ne
; CHECK-NEXT:    ret
   %srl = lshr i128 %a, 17
   %cmp = icmp ne i128 %srl, 0
   ret i1 %cmp
}

define i1 @opt_setcc_shl_eq_zero(i128 %a) nounwind {
; CHECK-LABEL: opt_setcc_shl_eq_zero:
; CHECK:       // %bb.0:
; CHECK-NEXT:    extr x8, x1, x0, #47
; CHECK-NEXT:    orr x8, x8, x0, lsl #17
; CHECK-NEXT:    cmp x8, #0
; CHECK-NEXT:    cset w0, eq
; CHECK-NEXT:    ret
   %shl = shl i128 %a, 17
   %cmp = icmp eq i128 %shl, 0
   ret i1 %cmp
}

define i1 @opt_setcc_shl_ne_zero(i128 %a) nounwind {
; CHECK-LABEL: opt_setcc_shl_ne_zero:
; CHECK:       // %bb.0:
; CHECK-NEXT:    extr x8, x1, x0, #47
; CHECK-NEXT:    orr x8, x8, x0, lsl #17
; CHECK-NEXT:    cmp x8, #0
; CHECK-NEXT:    cset w0, ne
; CHECK-NEXT:    ret
   %shl = shl i128 %a, 17
   %cmp = icmp ne i128 %shl, 0
   ret i1 %cmp
}

; Negative test: optimization should not be applied if shift has multiple users.
define i1 @opt_setcc_shl_eq_zero_multiple_shl_users(i128 %a) nounwind {
; CHECK-LABEL: opt_setcc_shl_eq_zero_multiple_shl_users:
; CHECK:       // %bb.0:
; CHECK-NEXT:    stp x30, x19, [sp, #-16]! // 16-byte Folded Spill
; CHECK-NEXT:    extr x1, x1, x0, #47
; CHECK-NEXT:    lsl x0, x0, #17
; CHECK-NEXT:    orr x8, x0, x1
; CHECK-NEXT:    cmp x8, #0
; CHECK-NEXT:    cset w19, eq
; CHECK-NEXT:    bl use
; CHECK-NEXT:    mov w0, w19
; CHECK-NEXT:    ldp x30, x19, [sp], #16 // 16-byte Folded Reload
; CHECK-NEXT:    ret
   %shl = shl i128 %a, 17
   %cmp = icmp eq i128 %shl, 0
   call void @use(i128 %shl)
   ret i1 %cmp
}

; Check that optimization is applied to DAG having appropriate shape
; even if there were no actual shift's expansion.
define i1 @opt_setcc_expanded_shl_correct_shifts(i64 %a, i64 %b) nounwind {
; CHECK-LABEL: opt_setcc_expanded_shl_correct_shifts:
; CHECK:       // %bb.0:
; CHECK-NEXT:    extr x8, x0, x1, #47
; CHECK-NEXT:    orr x8, x8, x1, lsl #17
; CHECK-NEXT:    cmp x8, #0
; CHECK-NEXT:    cset w0, eq
; CHECK-NEXT:    ret
  %shl.a = shl i64 %a, 17
  %srl.b = lshr i64 %b, 47
  %or.0 = or i64 %shl.a, %srl.b
  %shl.b = shl i64 %b, 17
  %or.1 = or i64 %or.0, %shl.b
  %cmp = icmp eq i64 %or.1, 0
  ret i1 %cmp
}

; Negative test: optimization should not be applied as
; constants used in shifts do not match.
define i1 @opt_setcc_expanded_shl_wrong_shifts(i64 %a, i64 %b) nounwind {
; CHECK-LABEL: opt_setcc_expanded_shl_wrong_shifts:
; CHECK:       // %bb.0:
; CHECK-NEXT:    extr x8, x0, x1, #47
; CHECK-NEXT:    orr x8, x8, x1, lsl #18
; CHECK-NEXT:    cmp x8, #0
; CHECK-NEXT:    cset w0, eq
; CHECK-NEXT:    ret
  %shl.a = shl i64 %a, 17
  %srl.b = lshr i64 %b, 47
  %or.0 = or i64 %shl.a, %srl.b
  %shl.b = shl i64 %b, 18
  %or.1 = or i64 %or.0, %shl.b
  %cmp = icmp eq i64 %or.1, 0
  ret i1 %cmp
}

define i1 @opt_setcc_shl_ne_zero_i256(i256 %a) nounwind {
; CHECK-LABEL: opt_setcc_shl_ne_zero_i256:
; CHECK:       // %bb.0:
; CHECK-NEXT:    extr x8, x3, x2, #47
; CHECK-NEXT:    extr x9, x2, x1, #47
; CHECK-NEXT:    extr x10, x1, x0, #47
; CHECK-NEXT:    orr x9, x9, x0, lsl #17
; CHECK-NEXT:    orr x8, x10, x8
; CHECK-NEXT:    orr x8, x9, x8
; CHECK-NEXT:    cmp x8, #0
; CHECK-NEXT:    cset w0, ne
; CHECK-NEXT:    ret
  %shl = shl i256 %a, 17
  %cmp = icmp ne i256 %shl, 0
  ret i1 %cmp
}

declare void @use(i128 %a)
