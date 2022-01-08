; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s

declare dso_local void @bar()

define void @foo(i64*) {
; CHECK-LABEL: foo:
; CHECK:       # %bb.0: # %start
; CHECK-NEXT:    movl (%rdi), %eax
; CHECK-NEXT:    andl $6, %eax
; CHECK-NEXT:    cmpl $4, %eax
; CHECK-NEXT:    jne .LBB0_2
; CHECK-NEXT:  # %bb.1: # %bb1
; CHECK-NEXT:    retq
; CHECK-NEXT:  .LBB0_2: # %bb2.i
; CHECK-NEXT:    jmp bar # TAILCALL
start:
  %1 = load i64, i64* %0, align 8, !range !0
  %2 = and i64 %1, 6
  %3 = icmp eq i64 %2, 4
  br i1 %3, label %bb1, label %bb2.i

bb1:                                              ; preds = %bb2.i, %start
  ret void

bb2.i:                                            ; preds = %start
  tail call fastcc void @bar()
  br label %bb1
}

!0 = !{i64 0, i64 6}
