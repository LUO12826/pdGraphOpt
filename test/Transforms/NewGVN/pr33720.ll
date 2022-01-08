; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -newgvn -S %s | FileCheck %s

@f = external local_unnamed_addr global i64
@b = external local_unnamed_addr global i64
@e = external local_unnamed_addr global i64

define void @patatino() {
; CHECK-LABEL: @patatino(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 undef, label [[IF_END24:%.*]], label [[FOR_COND16:%.*]]
; CHECK:       for.cond2thread-pre-split:
; CHECK-NEXT:    br i1 false, label [[FOR_BODY:%.*]], label [[FOR_COND8_PREHEADER:%.*]]
; CHECK:       for.cond8.preheader:
; CHECK-NEXT:    br i1 undef, label [[L1:%.*]], label [[FOR_COND11THREAD_PRE_SPLIT_LR_PH:%.*]]
; CHECK:       for.cond11thread-pre-split.lr.ph:
; CHECK-NEXT:    br label [[L1]]
; CHECK:       for.body:
; CHECK-NEXT:    [[CMP3:%.*]] = icmp ne i64 [[K_2:%.*]], 3
; CHECK-NEXT:    [[CONV4:%.*]] = zext i1 [[CMP3]] to i64
; CHECK-NEXT:    [[TMP0:%.*]] = load i64, i64* @f, align 4
; CHECK-NEXT:    [[OR:%.*]] = or i64 [[TMP0]], [[CONV4]]
; CHECK-NEXT:    store i64 [[OR]], i64* @f, align 4
; CHECK-NEXT:    [[TOBOOL7:%.*]] = icmp ne i64 [[K_2]], 0
; CHECK-NEXT:    br i1 [[TOBOOL7]], label [[FOR_COND2THREAD_PRE_SPLIT:%.*]], label [[LOR_RHS:%.*]]
; CHECK:       lor.rhs:
; CHECK-NEXT:    store i64 1, i64* @b, align 8
; CHECK-NEXT:    br label [[FOR_COND2THREAD_PRE_SPLIT]]
; CHECK:       l1:
; CHECK-NEXT:    [[K_2]] = phi i64 [ undef, [[L1_PREHEADER:%.*]] ], [ 15, [[FOR_COND8_PREHEADER]] ], [ 5, [[FOR_COND11THREAD_PRE_SPLIT_LR_PH]] ]
; CHECK-NEXT:    store i64 7, i64* [[J_3:%.*]], align 4
; CHECK-NEXT:    br label [[FOR_BODY]]
; CHECK:       for.cond16:
; CHECK-NEXT:    [[J_0:%.*]] = phi i64* [ @f, [[ENTRY:%.*]] ], [ poison, [[FOR_COND20:%.*]] ], [ @e, [[FOR_COND16]] ]
; CHECK-NEXT:    br i1 undef, label [[FOR_COND20]], label [[FOR_COND16]]
; CHECK:       for.cond20:
; CHECK-NEXT:    [[J_2:%.*]] = phi i64* [ [[J_0]], [[FOR_COND16]] ], [ poison, [[IF_END24]] ]
; CHECK-NEXT:    br i1 true, label [[IF_END24]], label [[FOR_COND16]]
; CHECK:       if.end24:
; CHECK-NEXT:    [[J_3]] = phi i64* [ [[J_2]], [[FOR_COND20]] ], [ undef, [[ENTRY]] ]
; CHECK-NEXT:    br i1 false, label [[FOR_COND20]], label [[L1_PREHEADER]]
; CHECK:       l1.preheader:
; CHECK-NEXT:    br label [[L1]]
;
entry:
  br i1 undef, label %if.end24, label %for.cond16

for.cond2thread-pre-split:
  br i1 false, label %for.body, label %for.cond8.preheader

for.cond8.preheader:
  br i1 undef, label %l1, label %for.cond11thread-pre-split.lr.ph

for.cond11thread-pre-split.lr.ph:
  br label %l1

for.body:
  %k.031 = phi i64 [ %k.2, %l1 ], [ 15, %for.cond2thread-pre-split ]
  %cmp3 = icmp ne i64 %k.031, 3
  %conv4 = zext i1 %cmp3 to i64
  %0 = load i64, i64* @f
  %or = or i64 %0, %conv4
  store i64 %or, i64* @f
  %tobool7 = icmp ne i64 %k.031, 0
  %or.cond = or i1 %tobool7, false
  br i1 %or.cond, label %for.cond2thread-pre-split, label %lor.rhs

lor.rhs:
  store i64 1, i64* @b, align 8
  br label %for.cond2thread-pre-split

l1:
  %k.2 = phi i64 [ undef, %l1.preheader ], [ 15, %for.cond8.preheader ], [ 5, %for.cond11thread-pre-split.lr.ph ]
  store i64 7, i64* %j.3
  br label %for.body

for.cond16:
  %j.0 = phi i64* [ @f, %entry ], [ %j.2, %for.cond20 ], [ @e, %for.cond16 ]
  br i1 undef, label %for.cond20, label %for.cond16

for.cond20:
  %j.2 = phi i64* [ %j.0, %for.cond16 ], [ %j.3, %if.end24 ]
  br i1 true, label %if.end24, label %for.cond16

if.end24:
  %j.3 = phi i64* [ %j.2, %for.cond20 ], [ undef, %entry ]
  br i1 false, label %for.cond20, label %l1.preheader

l1.preheader:
  br label %l1
}
