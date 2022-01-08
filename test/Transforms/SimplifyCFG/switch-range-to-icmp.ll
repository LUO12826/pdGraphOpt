; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s

declare i32 @f(i32)

define i32 @basic(i32 %x) {
; CHECK-LABEL: @basic(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[X_OFF:%.*]] = add i32 [[X:%.*]], -5
; CHECK-NEXT:    [[SWITCH:%.*]] = icmp ult i32 [[X_OFF]], 3
; CHECK-NEXT:    br i1 [[SWITCH]], label [[A:%.*]], label [[DEFAULT:%.*]]
; CHECK:       common.ret:
; CHECK-NEXT:    [[COMMON_RET_OP:%.*]] = phi i32 [ [[TMP0:%.*]], [[DEFAULT]] ], [ [[TMP1:%.*]], [[A]] ]
; CHECK-NEXT:    ret i32 [[COMMON_RET_OP]]
; CHECK:       default:
; CHECK-NEXT:    [[TMP0]] = call i32 @f(i32 0)
; CHECK-NEXT:    br label [[COMMON_RET:%.*]]
; CHECK:       a:
; CHECK-NEXT:    [[TMP1]] = call i32 @f(i32 1)
; CHECK-NEXT:    br label [[COMMON_RET]]
;

entry:
  switch i32 %x, label %default [
  i32 5, label %a
  i32 6, label %a
  i32 7, label %a
  ]
default:
  %0 = call i32 @f(i32 0)
  ret i32 %0
a:
  %1 = call i32 @f(i32 1)
  ret i32 %1
}


define i32 @unreachable(i32 %x) {
; CHECK-LABEL: @unreachable(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[X_OFF:%.*]] = add i32 [[X:%.*]], -5
; CHECK-NEXT:    [[SWITCH:%.*]] = icmp ult i32 [[X_OFF]], 3
; CHECK-NEXT:    br i1 [[SWITCH]], label [[A:%.*]], label [[B:%.*]]
; CHECK:       common.ret:
; CHECK-NEXT:    [[COMMON_RET_OP:%.*]] = phi i32 [ [[TMP0:%.*]], [[A]] ], [ [[TMP1:%.*]], [[B]] ]
; CHECK-NEXT:    ret i32 [[COMMON_RET_OP]]
; CHECK:       a:
; CHECK-NEXT:    [[TMP0]] = call i32 @f(i32 0)
; CHECK-NEXT:    br label [[COMMON_RET:%.*]]
; CHECK:       b:
; CHECK-NEXT:    [[TMP1]] = call i32 @f(i32 1)
; CHECK-NEXT:    br label [[COMMON_RET]]
;

entry:
  switch i32 %x, label %unreachable [
  i32 5, label %a
  i32 6, label %a
  i32 7, label %a
  i32 10, label %b
  i32 20, label %b
  i32 30, label %b
  i32 40, label %b
  ]
unreachable:
  unreachable
a:
  %0 = call i32 @f(i32 0)
  ret i32 %0
b:
  %1 = call i32 @f(i32 1)
  ret i32 %1
}


define i32 @unreachable2(i32 %x) {
; CHECK-LABEL: @unreachable2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[X_OFF:%.*]] = add i32 [[X:%.*]], -5
; CHECK-NEXT:    [[SWITCH:%.*]] = icmp ult i32 [[X_OFF]], 3
; CHECK-NEXT:    br i1 [[SWITCH]], label [[A:%.*]], label [[B:%.*]]
; CHECK:       common.ret:
; CHECK-NEXT:    [[COMMON_RET_OP:%.*]] = phi i32 [ [[TMP0:%.*]], [[A]] ], [ [[TMP1:%.*]], [[B]] ]
; CHECK-NEXT:    ret i32 [[COMMON_RET_OP]]
; CHECK:       a:
; CHECK-NEXT:    [[TMP0]] = call i32 @f(i32 0)
; CHECK-NEXT:    br label [[COMMON_RET:%.*]]
; CHECK:       b:
; CHECK-NEXT:    [[TMP1]] = call i32 @f(i32 1)
; CHECK-NEXT:    br label [[COMMON_RET]]
;

entry:
  ; Note: folding the most popular case destination into the default
  ; would prevent switch-to-icmp here.
  switch i32 %x, label %unreachable [
  i32 5, label %a
  i32 6, label %a
  i32 7, label %a
  i32 10, label %b
  i32 20, label %b
  ]
unreachable:
  unreachable
a:
  %0 = call i32 @f(i32 0)
  ret i32 %0
b:
  %1 = call i32 @f(i32 1)
  ret i32 %1
}

; This would crash because we did not clean up the
; default block of the switch before removing the switch.

define void @PR42737(i32* %a, i1 %c) {
; CHECK-LABEL: @PR42737(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    unreachable
;
entry:
  br i1 %c, label %switch, label %else

else:
  store i32 2, i32* %a
  br label %switch

switch:
  %cleanup.dest1 = phi i32 [ 0, %else ], [ 3, %entry ]
  switch i32 %cleanup.dest1, label %unreach1 [
  i32 0, label %cleanup1
  i32 3, label %cleanup2
  ]

cleanup1:
  br label %unreach2

cleanup2:
  br label %unreach2

unreach1:
  %phi2 = phi i32 [ %cleanup.dest1, %switch ]
  unreachable

unreach2:
  unreachable
}
