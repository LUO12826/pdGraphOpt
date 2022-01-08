; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --function-signature --scrub-attributes
; RUN: opt -openmp-opt-cgscc -S < %s | FileCheck %s
; RUN: opt -passes=openmp-opt-cgscc -S < %s | FileCheck %s

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64"

%struct.ident_t = type { i32, i32, i32, i32, i8* }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @0, i32 0, i32 0) }, align 8
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 1, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @0, i32 0, i32 0) }, align 8
@__omp_offloading_50_a3e09bf8_foo_l2_exec_mode = weak constant i8 0
@llvm.compiler.used = appending global [1 x i8*] [i8* @__omp_offloading_50_a3e09bf8_foo_l2_exec_mode], section "llvm.metadata"

declare void @use(i32)

define weak void @__omp_offloading_50_a3e09bf8_foo_l2() #0 {
; CHECK-LABEL: define {{[^@]+}}@__omp_offloading_50_a3e09bf8_foo_l2
; CHECK-SAME: () #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CAPTURED_VARS_ADDRS:%.*]] = alloca [0 x i8*], align 8
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__kmpc_target_init(%struct.ident_t* @[[GLOB1:[0-9]+]], i8 2, i1 false, i1 true)
; CHECK-NEXT:    [[TMP1:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB2:[0-9]+]])
; CHECK-NEXT:    [[EXEC_USER_CODE:%.*]] = icmp eq i32 [[TMP0]], -1
; CHECK-NEXT:    br i1 [[EXEC_USER_CODE]], label [[USER_CODE_ENTRY:%.*]], label [[WORKER_EXIT:%.*]]
; CHECK:       user_code.entry:
; CHECK-NEXT:    call void @__kmpc_target_deinit(%struct.ident_t* @[[GLOB1]], i8 2, i1 true)
; CHECK-NEXT:    ret void
; CHECK:       worker.exit:
; CHECK-NEXT:    ret void
;
entry:
  %captured_vars_addrs = alloca [0 x i8*], align 8
  %0 = call i32 @__kmpc_target_init(%struct.ident_t* @1, i8 2, i1 false, i1 true)
  %exec_user_code = icmp eq i32 %0, -1
  br i1 %exec_user_code, label %user_code.entry, label %worker.exit

user_code.entry:                                  ; preds = %entry
  %1 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @2)
  %2 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @2)
  call void @__kmpc_target_deinit(%struct.ident_t* @1, i8 2, i1 true)
  ret void

worker.exit:                                      ; preds = %entry
  ret void
}

declare i32 @__kmpc_target_init(%struct.ident_t*, i8, i1, i1)

declare i32 @__kmpc_global_thread_num(%struct.ident_t*) #1

declare void @__kmpc_target_deinit(%struct.ident_t*, i8, i1)

attributes #0 = { convergent noinline norecurse nounwind "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+ptx32,+sm_20" }
attributes #1 = { nounwind }

!omp_offload.info = !{!0}
!nvvm.annotations = !{!1}
!llvm.module.flags = !{!2, !3, !4}

!0 = !{i32 0, i32 80, i32 -1545561096, !"foo", i32 2, i32 0}
!1 = !{void ()* @__omp_offloading_50_a3e09bf8_foo_l2, !"kernel", i32 1}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 7, !"openmp", i32 50}
!4 = !{i32 7, !"openmp-device", i32 50}
