; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -loop-idiom -mtriple=riscv32 -mattr=+experimental-zbb -S < %s | FileCheck %s --check-prefixes=CPOP
; RUN: opt -loop-idiom -mtriple=riscv64 -mattr=+experimental-zbb -S < %s | FileCheck %s --check-prefixes=CPOP
; RUN: opt -loop-idiom -mtriple=riscv32 -S < %s | FileCheck %s --check-prefixes=NOCPOP
; RUN: opt -loop-idiom -mtriple=riscv64 -S < %s | FileCheck %s --check-prefixes=NOCPOP

; Mostly copied from AMDGPU version.

;To recognize this pattern:
;int popcount(unsigned long long a) {
;    int c = 0;
;    while (a) {
;        c++;
;        a &= a - 1;
;    }
;    return c;
;}
;

define i32 @popcount_i64(i64 %a) nounwind uwtable readnone ssp {
; CPOP-LABEL: @popcount_i64(
; CPOP-NEXT:  entry:
; CPOP-NEXT:    [[TMP0:%.*]] = call i64 @llvm.ctpop.i64(i64 [[A:%.*]])
; CPOP-NEXT:    [[TMP1:%.*]] = trunc i64 [[TMP0]] to i32
; CPOP-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[TMP1]], 0
; CPOP-NEXT:    br i1 [[TMP2]], label [[WHILE_END:%.*]], label [[WHILE_BODY_PREHEADER:%.*]]
; CPOP:       while.body.preheader:
; CPOP-NEXT:    br label [[WHILE_BODY:%.*]]
; CPOP:       while.body:
; CPOP-NEXT:    [[TCPHI:%.*]] = phi i32 [ [[TMP1]], [[WHILE_BODY_PREHEADER]] ], [ [[TCDEC:%.*]], [[WHILE_BODY]] ]
; CPOP-NEXT:    [[C_05:%.*]] = phi i32 [ [[INC:%.*]], [[WHILE_BODY]] ], [ 0, [[WHILE_BODY_PREHEADER]] ]
; CPOP-NEXT:    [[A_ADDR_04:%.*]] = phi i64 [ [[AND:%.*]], [[WHILE_BODY]] ], [ [[A]], [[WHILE_BODY_PREHEADER]] ]
; CPOP-NEXT:    [[INC]] = add nsw i32 [[C_05]], 1
; CPOP-NEXT:    [[SUB:%.*]] = add i64 [[A_ADDR_04]], -1
; CPOP-NEXT:    [[AND]] = and i64 [[SUB]], [[A_ADDR_04]]
; CPOP-NEXT:    [[TCDEC]] = sub nsw i32 [[TCPHI]], 1
; CPOP-NEXT:    [[TOBOOL:%.*]] = icmp sle i32 [[TCDEC]], 0
; CPOP-NEXT:    br i1 [[TOBOOL]], label [[WHILE_END_LOOPEXIT:%.*]], label [[WHILE_BODY]]
; CPOP:       while.end.loopexit:
; CPOP-NEXT:    [[INC_LCSSA:%.*]] = phi i32 [ [[TMP1]], [[WHILE_BODY]] ]
; CPOP-NEXT:    br label [[WHILE_END]]
; CPOP:       while.end:
; CPOP-NEXT:    [[C_0_LCSSA:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[INC_LCSSA]], [[WHILE_END_LOOPEXIT]] ]
; CPOP-NEXT:    ret i32 [[C_0_LCSSA]]
;
; NOCPOP-LABEL: @popcount_i64(
; NOCPOP-NEXT:  entry:
; NOCPOP-NEXT:    [[TOBOOL3:%.*]] = icmp eq i64 [[A:%.*]], 0
; NOCPOP-NEXT:    br i1 [[TOBOOL3]], label [[WHILE_END:%.*]], label [[WHILE_BODY_PREHEADER:%.*]]
; NOCPOP:       while.body.preheader:
; NOCPOP-NEXT:    br label [[WHILE_BODY:%.*]]
; NOCPOP:       while.body:
; NOCPOP-NEXT:    [[C_05:%.*]] = phi i32 [ [[INC:%.*]], [[WHILE_BODY]] ], [ 0, [[WHILE_BODY_PREHEADER]] ]
; NOCPOP-NEXT:    [[A_ADDR_04:%.*]] = phi i64 [ [[AND:%.*]], [[WHILE_BODY]] ], [ [[A]], [[WHILE_BODY_PREHEADER]] ]
; NOCPOP-NEXT:    [[INC]] = add nsw i32 [[C_05]], 1
; NOCPOP-NEXT:    [[SUB:%.*]] = add i64 [[A_ADDR_04]], -1
; NOCPOP-NEXT:    [[AND]] = and i64 [[SUB]], [[A_ADDR_04]]
; NOCPOP-NEXT:    [[TOBOOL:%.*]] = icmp eq i64 [[AND]], 0
; NOCPOP-NEXT:    br i1 [[TOBOOL]], label [[WHILE_END_LOOPEXIT:%.*]], label [[WHILE_BODY]]
; NOCPOP:       while.end.loopexit:
; NOCPOP-NEXT:    [[INC_LCSSA:%.*]] = phi i32 [ [[INC]], [[WHILE_BODY]] ]
; NOCPOP-NEXT:    br label [[WHILE_END]]
; NOCPOP:       while.end:
; NOCPOP-NEXT:    [[C_0_LCSSA:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[INC_LCSSA]], [[WHILE_END_LOOPEXIT]] ]
; NOCPOP-NEXT:    ret i32 [[C_0_LCSSA]]
;
entry:
  %tobool3 = icmp eq i64 %a, 0
  br i1 %tobool3, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %c.05 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %a.addr.04 = phi i64 [ %and, %while.body ], [ %a, %entry ]
  %inc = add nsw i32 %c.05, 1
  %sub = add i64 %a.addr.04, -1
  %and = and i64 %sub, %a.addr.04
  %tobool = icmp eq i64 %and, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  %c.0.lcssa = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  ret i32 %c.0.lcssa
}

define i32 @popcount_i32(i32 %a) nounwind uwtable readnone ssp {
; CPOP-LABEL: @popcount_i32(
; CPOP-NEXT:  entry:
; CPOP-NEXT:    [[TMP0:%.*]] = call i32 @llvm.ctpop.i32(i32 [[A:%.*]])
; CPOP-NEXT:    [[TMP1:%.*]] = icmp eq i32 [[TMP0]], 0
; CPOP-NEXT:    br i1 [[TMP1]], label [[WHILE_END:%.*]], label [[WHILE_BODY_PREHEADER:%.*]]
; CPOP:       while.body.preheader:
; CPOP-NEXT:    br label [[WHILE_BODY:%.*]]
; CPOP:       while.body:
; CPOP-NEXT:    [[TCPHI:%.*]] = phi i32 [ [[TMP0]], [[WHILE_BODY_PREHEADER]] ], [ [[TCDEC:%.*]], [[WHILE_BODY]] ]
; CPOP-NEXT:    [[C_05:%.*]] = phi i32 [ [[INC:%.*]], [[WHILE_BODY]] ], [ 0, [[WHILE_BODY_PREHEADER]] ]
; CPOP-NEXT:    [[A_ADDR_04:%.*]] = phi i32 [ [[AND:%.*]], [[WHILE_BODY]] ], [ [[A]], [[WHILE_BODY_PREHEADER]] ]
; CPOP-NEXT:    [[INC]] = add nsw i32 [[C_05]], 1
; CPOP-NEXT:    [[SUB:%.*]] = add i32 [[A_ADDR_04]], -1
; CPOP-NEXT:    [[AND]] = and i32 [[SUB]], [[A_ADDR_04]]
; CPOP-NEXT:    [[TCDEC]] = sub nsw i32 [[TCPHI]], 1
; CPOP-NEXT:    [[TOBOOL:%.*]] = icmp sle i32 [[TCDEC]], 0
; CPOP-NEXT:    br i1 [[TOBOOL]], label [[WHILE_END_LOOPEXIT:%.*]], label [[WHILE_BODY]]
; CPOP:       while.end.loopexit:
; CPOP-NEXT:    [[INC_LCSSA:%.*]] = phi i32 [ [[TMP0]], [[WHILE_BODY]] ]
; CPOP-NEXT:    br label [[WHILE_END]]
; CPOP:       while.end:
; CPOP-NEXT:    [[C_0_LCSSA:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[INC_LCSSA]], [[WHILE_END_LOOPEXIT]] ]
; CPOP-NEXT:    ret i32 [[C_0_LCSSA]]
;
; NOCPOP-LABEL: @popcount_i32(
; NOCPOP-NEXT:  entry:
; NOCPOP-NEXT:    [[TOBOOL3:%.*]] = icmp eq i32 [[A:%.*]], 0
; NOCPOP-NEXT:    br i1 [[TOBOOL3]], label [[WHILE_END:%.*]], label [[WHILE_BODY_PREHEADER:%.*]]
; NOCPOP:       while.body.preheader:
; NOCPOP-NEXT:    br label [[WHILE_BODY:%.*]]
; NOCPOP:       while.body:
; NOCPOP-NEXT:    [[C_05:%.*]] = phi i32 [ [[INC:%.*]], [[WHILE_BODY]] ], [ 0, [[WHILE_BODY_PREHEADER]] ]
; NOCPOP-NEXT:    [[A_ADDR_04:%.*]] = phi i32 [ [[AND:%.*]], [[WHILE_BODY]] ], [ [[A]], [[WHILE_BODY_PREHEADER]] ]
; NOCPOP-NEXT:    [[INC]] = add nsw i32 [[C_05]], 1
; NOCPOP-NEXT:    [[SUB:%.*]] = add i32 [[A_ADDR_04]], -1
; NOCPOP-NEXT:    [[AND]] = and i32 [[SUB]], [[A_ADDR_04]]
; NOCPOP-NEXT:    [[TOBOOL:%.*]] = icmp eq i32 [[AND]], 0
; NOCPOP-NEXT:    br i1 [[TOBOOL]], label [[WHILE_END_LOOPEXIT:%.*]], label [[WHILE_BODY]]
; NOCPOP:       while.end.loopexit:
; NOCPOP-NEXT:    [[INC_LCSSA:%.*]] = phi i32 [ [[INC]], [[WHILE_BODY]] ]
; NOCPOP-NEXT:    br label [[WHILE_END]]
; NOCPOP:       while.end:
; NOCPOP-NEXT:    [[C_0_LCSSA:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[INC_LCSSA]], [[WHILE_END_LOOPEXIT]] ]
; NOCPOP-NEXT:    ret i32 [[C_0_LCSSA]]
;
entry:
  %tobool3 = icmp eq i32 %a, 0
  br i1 %tobool3, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %c.05 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %a.addr.04 = phi i32 [ %and, %while.body ], [ %a, %entry ]
  %inc = add nsw i32 %c.05, 1
  %sub = add i32 %a.addr.04, -1
  %and = and i32 %sub, %a.addr.04
  %tobool = icmp eq i32 %and, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  %c.0.lcssa = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  ret i32 %c.0.lcssa
}

define i32 @popcount_i128(i128 %a) nounwind uwtable readnone ssp {
; CPOP-LABEL: @popcount_i128(
; CPOP-NEXT:  entry:
; CPOP-NEXT:    [[TMP0:%.*]] = call i128 @llvm.ctpop.i128(i128 [[A:%.*]])
; CPOP-NEXT:    [[TMP1:%.*]] = trunc i128 [[TMP0]] to i32
; CPOP-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[TMP1]], 0
; CPOP-NEXT:    br i1 [[TMP2]], label [[WHILE_END:%.*]], label [[WHILE_BODY_PREHEADER:%.*]]
; CPOP:       while.body.preheader:
; CPOP-NEXT:    br label [[WHILE_BODY:%.*]]
; CPOP:       while.body:
; CPOP-NEXT:    [[TCPHI:%.*]] = phi i32 [ [[TMP1]], [[WHILE_BODY_PREHEADER]] ], [ [[TCDEC:%.*]], [[WHILE_BODY]] ]
; CPOP-NEXT:    [[C_05:%.*]] = phi i32 [ [[INC:%.*]], [[WHILE_BODY]] ], [ 0, [[WHILE_BODY_PREHEADER]] ]
; CPOP-NEXT:    [[A_ADDR_04:%.*]] = phi i128 [ [[AND:%.*]], [[WHILE_BODY]] ], [ [[A]], [[WHILE_BODY_PREHEADER]] ]
; CPOP-NEXT:    [[INC]] = add nsw i32 [[C_05]], 1
; CPOP-NEXT:    [[SUB:%.*]] = add i128 [[A_ADDR_04]], -1
; CPOP-NEXT:    [[AND]] = and i128 [[SUB]], [[A_ADDR_04]]
; CPOP-NEXT:    [[TCDEC]] = sub nsw i32 [[TCPHI]], 1
; CPOP-NEXT:    [[TOBOOL:%.*]] = icmp sle i32 [[TCDEC]], 0
; CPOP-NEXT:    br i1 [[TOBOOL]], label [[WHILE_END_LOOPEXIT:%.*]], label [[WHILE_BODY]]
; CPOP:       while.end.loopexit:
; CPOP-NEXT:    [[INC_LCSSA:%.*]] = phi i32 [ [[TMP1]], [[WHILE_BODY]] ]
; CPOP-NEXT:    br label [[WHILE_END]]
; CPOP:       while.end:
; CPOP-NEXT:    [[C_0_LCSSA:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[INC_LCSSA]], [[WHILE_END_LOOPEXIT]] ]
; CPOP-NEXT:    ret i32 [[C_0_LCSSA]]
;
; NOCPOP-LABEL: @popcount_i128(
; NOCPOP-NEXT:  entry:
; NOCPOP-NEXT:    [[TOBOOL3:%.*]] = icmp eq i128 [[A:%.*]], 0
; NOCPOP-NEXT:    br i1 [[TOBOOL3]], label [[WHILE_END:%.*]], label [[WHILE_BODY_PREHEADER:%.*]]
; NOCPOP:       while.body.preheader:
; NOCPOP-NEXT:    br label [[WHILE_BODY:%.*]]
; NOCPOP:       while.body:
; NOCPOP-NEXT:    [[C_05:%.*]] = phi i32 [ [[INC:%.*]], [[WHILE_BODY]] ], [ 0, [[WHILE_BODY_PREHEADER]] ]
; NOCPOP-NEXT:    [[A_ADDR_04:%.*]] = phi i128 [ [[AND:%.*]], [[WHILE_BODY]] ], [ [[A]], [[WHILE_BODY_PREHEADER]] ]
; NOCPOP-NEXT:    [[INC]] = add nsw i32 [[C_05]], 1
; NOCPOP-NEXT:    [[SUB:%.*]] = add i128 [[A_ADDR_04]], -1
; NOCPOP-NEXT:    [[AND]] = and i128 [[SUB]], [[A_ADDR_04]]
; NOCPOP-NEXT:    [[TOBOOL:%.*]] = icmp eq i128 [[AND]], 0
; NOCPOP-NEXT:    br i1 [[TOBOOL]], label [[WHILE_END_LOOPEXIT:%.*]], label [[WHILE_BODY]]
; NOCPOP:       while.end.loopexit:
; NOCPOP-NEXT:    [[INC_LCSSA:%.*]] = phi i32 [ [[INC]], [[WHILE_BODY]] ]
; NOCPOP-NEXT:    br label [[WHILE_END]]
; NOCPOP:       while.end:
; NOCPOP-NEXT:    [[C_0_LCSSA:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[INC_LCSSA]], [[WHILE_END_LOOPEXIT]] ]
; NOCPOP-NEXT:    ret i32 [[C_0_LCSSA]]
;
entry:
  %tobool3 = icmp eq i128 %a, 0
  br i1 %tobool3, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %c.05 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %a.addr.04 = phi i128 [ %and, %while.body ], [ %a, %entry ]
  %inc = add nsw i32 %c.05, 1
  %sub = add i128 %a.addr.04, -1
  %and = and i128 %sub, %a.addr.04
  %tobool = icmp eq i128 %and, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  %c.0.lcssa = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  ret i32 %c.0.lcssa
}

; To recognize this pattern:
;int popcount(unsigned long long a, int mydata1, int mydata2) {
;    int c = 0;
;    while (a) {
;        c++;
;        a &= a - 1;
;        mydata1 *= c;
;        mydata2 *= (int)a;
;    }
;    return c + mydata1 + mydata2;
;}

define i32 @popcount2(i64 %a, i32 %mydata1, i32 %mydata2) nounwind uwtable readnone ssp {
; CPOP-LABEL: @popcount2(
; CPOP-NEXT:  entry:
; CPOP-NEXT:    [[TMP0:%.*]] = call i64 @llvm.ctpop.i64(i64 [[A:%.*]])
; CPOP-NEXT:    [[TMP1:%.*]] = trunc i64 [[TMP0]] to i32
; CPOP-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[TMP1]], 0
; CPOP-NEXT:    br i1 [[TMP2]], label [[WHILE_END:%.*]], label [[WHILE_BODY_PREHEADER:%.*]]
; CPOP:       while.body.preheader:
; CPOP-NEXT:    br label [[WHILE_BODY:%.*]]
; CPOP:       while.body:
; CPOP-NEXT:    [[TCPHI:%.*]] = phi i32 [ [[TMP1]], [[WHILE_BODY_PREHEADER]] ], [ [[TCDEC:%.*]], [[WHILE_BODY]] ]
; CPOP-NEXT:    [[C_013:%.*]] = phi i32 [ [[INC:%.*]], [[WHILE_BODY]] ], [ 0, [[WHILE_BODY_PREHEADER]] ]
; CPOP-NEXT:    [[MYDATA2_ADDR_012:%.*]] = phi i32 [ [[MUL1:%.*]], [[WHILE_BODY]] ], [ [[MYDATA2:%.*]], [[WHILE_BODY_PREHEADER]] ]
; CPOP-NEXT:    [[MYDATA1_ADDR_011:%.*]] = phi i32 [ [[MUL:%.*]], [[WHILE_BODY]] ], [ [[MYDATA1:%.*]], [[WHILE_BODY_PREHEADER]] ]
; CPOP-NEXT:    [[A_ADDR_010:%.*]] = phi i64 [ [[AND:%.*]], [[WHILE_BODY]] ], [ [[A]], [[WHILE_BODY_PREHEADER]] ]
; CPOP-NEXT:    [[INC]] = add nsw i32 [[C_013]], 1
; CPOP-NEXT:    [[SUB:%.*]] = add i64 [[A_ADDR_010]], -1
; CPOP-NEXT:    [[AND]] = and i64 [[SUB]], [[A_ADDR_010]]
; CPOP-NEXT:    [[MUL]] = mul nsw i32 [[INC]], [[MYDATA1_ADDR_011]]
; CPOP-NEXT:    [[CONV:%.*]] = trunc i64 [[AND]] to i32
; CPOP-NEXT:    [[MUL1]] = mul nsw i32 [[CONV]], [[MYDATA2_ADDR_012]]
; CPOP-NEXT:    [[TCDEC]] = sub nsw i32 [[TCPHI]], 1
; CPOP-NEXT:    [[TOBOOL:%.*]] = icmp sle i32 [[TCDEC]], 0
; CPOP-NEXT:    br i1 [[TOBOOL]], label [[WHILE_END_LOOPEXIT:%.*]], label [[WHILE_BODY]]
; CPOP:       while.end.loopexit:
; CPOP-NEXT:    [[INC_LCSSA:%.*]] = phi i32 [ [[TMP1]], [[WHILE_BODY]] ]
; CPOP-NEXT:    [[MUL_LCSSA:%.*]] = phi i32 [ [[MUL]], [[WHILE_BODY]] ]
; CPOP-NEXT:    [[MUL1_LCSSA:%.*]] = phi i32 [ [[MUL1]], [[WHILE_BODY]] ]
; CPOP-NEXT:    br label [[WHILE_END]]
; CPOP:       while.end:
; CPOP-NEXT:    [[C_0_LCSSA:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[INC_LCSSA]], [[WHILE_END_LOOPEXIT]] ]
; CPOP-NEXT:    [[MYDATA2_ADDR_0_LCSSA:%.*]] = phi i32 [ [[MYDATA2]], [[ENTRY]] ], [ [[MUL1_LCSSA]], [[WHILE_END_LOOPEXIT]] ]
; CPOP-NEXT:    [[MYDATA1_ADDR_0_LCSSA:%.*]] = phi i32 [ [[MYDATA1]], [[ENTRY]] ], [ [[MUL_LCSSA]], [[WHILE_END_LOOPEXIT]] ]
; CPOP-NEXT:    [[ADD:%.*]] = add i32 [[MYDATA2_ADDR_0_LCSSA]], [[MYDATA1_ADDR_0_LCSSA]]
; CPOP-NEXT:    [[ADD2:%.*]] = add i32 [[ADD]], [[C_0_LCSSA]]
; CPOP-NEXT:    ret i32 [[ADD2]]
;
; NOCPOP-LABEL: @popcount2(
; NOCPOP-NEXT:  entry:
; NOCPOP-NEXT:    [[TOBOOL9:%.*]] = icmp eq i64 [[A:%.*]], 0
; NOCPOP-NEXT:    br i1 [[TOBOOL9]], label [[WHILE_END:%.*]], label [[WHILE_BODY_PREHEADER:%.*]]
; NOCPOP:       while.body.preheader:
; NOCPOP-NEXT:    br label [[WHILE_BODY:%.*]]
; NOCPOP:       while.body:
; NOCPOP-NEXT:    [[C_013:%.*]] = phi i32 [ [[INC:%.*]], [[WHILE_BODY]] ], [ 0, [[WHILE_BODY_PREHEADER]] ]
; NOCPOP-NEXT:    [[MYDATA2_ADDR_012:%.*]] = phi i32 [ [[MUL1:%.*]], [[WHILE_BODY]] ], [ [[MYDATA2:%.*]], [[WHILE_BODY_PREHEADER]] ]
; NOCPOP-NEXT:    [[MYDATA1_ADDR_011:%.*]] = phi i32 [ [[MUL:%.*]], [[WHILE_BODY]] ], [ [[MYDATA1:%.*]], [[WHILE_BODY_PREHEADER]] ]
; NOCPOP-NEXT:    [[A_ADDR_010:%.*]] = phi i64 [ [[AND:%.*]], [[WHILE_BODY]] ], [ [[A]], [[WHILE_BODY_PREHEADER]] ]
; NOCPOP-NEXT:    [[INC]] = add nsw i32 [[C_013]], 1
; NOCPOP-NEXT:    [[SUB:%.*]] = add i64 [[A_ADDR_010]], -1
; NOCPOP-NEXT:    [[AND]] = and i64 [[SUB]], [[A_ADDR_010]]
; NOCPOP-NEXT:    [[MUL]] = mul nsw i32 [[INC]], [[MYDATA1_ADDR_011]]
; NOCPOP-NEXT:    [[CONV:%.*]] = trunc i64 [[AND]] to i32
; NOCPOP-NEXT:    [[MUL1]] = mul nsw i32 [[CONV]], [[MYDATA2_ADDR_012]]
; NOCPOP-NEXT:    [[TOBOOL:%.*]] = icmp eq i64 [[AND]], 0
; NOCPOP-NEXT:    br i1 [[TOBOOL]], label [[WHILE_END_LOOPEXIT:%.*]], label [[WHILE_BODY]]
; NOCPOP:       while.end.loopexit:
; NOCPOP-NEXT:    [[INC_LCSSA:%.*]] = phi i32 [ [[INC]], [[WHILE_BODY]] ]
; NOCPOP-NEXT:    [[MUL_LCSSA:%.*]] = phi i32 [ [[MUL]], [[WHILE_BODY]] ]
; NOCPOP-NEXT:    [[MUL1_LCSSA:%.*]] = phi i32 [ [[MUL1]], [[WHILE_BODY]] ]
; NOCPOP-NEXT:    br label [[WHILE_END]]
; NOCPOP:       while.end:
; NOCPOP-NEXT:    [[C_0_LCSSA:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[INC_LCSSA]], [[WHILE_END_LOOPEXIT]] ]
; NOCPOP-NEXT:    [[MYDATA2_ADDR_0_LCSSA:%.*]] = phi i32 [ [[MYDATA2]], [[ENTRY]] ], [ [[MUL1_LCSSA]], [[WHILE_END_LOOPEXIT]] ]
; NOCPOP-NEXT:    [[MYDATA1_ADDR_0_LCSSA:%.*]] = phi i32 [ [[MYDATA1]], [[ENTRY]] ], [ [[MUL_LCSSA]], [[WHILE_END_LOOPEXIT]] ]
; NOCPOP-NEXT:    [[ADD:%.*]] = add i32 [[MYDATA2_ADDR_0_LCSSA]], [[MYDATA1_ADDR_0_LCSSA]]
; NOCPOP-NEXT:    [[ADD2:%.*]] = add i32 [[ADD]], [[C_0_LCSSA]]
; NOCPOP-NEXT:    ret i32 [[ADD2]]
;
entry:
  %tobool9 = icmp eq i64 %a, 0
  br i1 %tobool9, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %c.013 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %mydata2.addr.012 = phi i32 [ %mul1, %while.body ], [ %mydata2, %entry ]
  %mydata1.addr.011 = phi i32 [ %mul, %while.body ], [ %mydata1, %entry ]
  %a.addr.010 = phi i64 [ %and, %while.body ], [ %a, %entry ]
  %inc = add nsw i32 %c.013, 1
  %sub = add i64 %a.addr.010, -1
  %and = and i64 %sub, %a.addr.010
  %mul = mul nsw i32 %inc, %mydata1.addr.011
  %conv = trunc i64 %and to i32
  %mul1 = mul nsw i32 %conv, %mydata2.addr.012
  %tobool = icmp eq i64 %and, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  %c.0.lcssa = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  %mydata2.addr.0.lcssa = phi i32 [ %mydata2, %entry ], [ %mul1, %while.body ]
  %mydata1.addr.0.lcssa = phi i32 [ %mydata1, %entry ], [ %mul, %while.body ]
  %add = add i32 %mydata2.addr.0.lcssa, %mydata1.addr.0.lcssa
  %add2 = add i32 %add, %c.0.lcssa
  ret i32 %add2
}
