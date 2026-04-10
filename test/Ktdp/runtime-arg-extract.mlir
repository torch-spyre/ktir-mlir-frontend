// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s"

// CHECK-LABEL:   func.func @extract_value(
// CHECK-SAME:                             %[[VAL_0:.*]]: !ktdp.runtime_arg<index, granularity=3, upperbound=300>) {
// CHECK-NEXT:     %[[VAL_1:.*]] = ktdp.runtime_arg_extract value from %[[VAL_0]] : <index, granularity=3, upperbound=300> -> index
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @extract_granularity(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !ktdp.runtime_arg<index, granularity=3, upperbound=300>) {
// CHECK-NEXT:     %[[VAL_1:.*]] = ktdp.runtime_arg_extract granularity from %[[VAL_0]] : <index, granularity=3, upperbound=300> -> index
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @extract_upperbound(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !ktdp.runtime_arg<index, granularity=3, upperbound=300>) {
// CHECK-NEXT:     %[[VAL_1:.*]] = ktdp.runtime_arg_extract upperbound from %[[VAL_0]] : <index, granularity=3, upperbound=300> -> index
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @extract_all(
// CHECK-SAME:                           %[[VAL_0:.*]]: !ktdp.runtime_arg<index, granularity=3, upperbound=300>) {
// CHECK-NEXT:     %[[VAL_1:.*]] = ktdp.runtime_arg_extract value from %[[VAL_0]] : <index, granularity=3, upperbound=300> -> index
// CHECK-NEXT:     %[[VAL_2:.*]] = ktdp.runtime_arg_extract granularity from %[[VAL_0]] : <index, granularity=3, upperbound=300> -> index
// CHECK-NEXT:     %[[VAL_3:.*]] = ktdp.runtime_arg_extract upperbound from %[[VAL_0]] : <index, granularity=3, upperbound=300> -> index
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @extract_value_only_granularity(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !ktdp.runtime_arg<index, granularity=4>) {
// CHECK-NEXT:     %[[VAL_1:.*]] = ktdp.runtime_arg_extract value from %[[VAL_0]] : <index, granularity=4> -> index
// CHECK-NEXT:     %[[VAL_2:.*]] = ktdp.runtime_arg_extract granularity from %[[VAL_0]] : <index, granularity=4> -> index
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @extract_i32_value(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !ktdp.runtime_arg<i32, granularity=8>) {
// CHECK-NEXT:     %[[VAL_1:.*]] = ktdp.runtime_arg_extract value from %[[VAL_0]] : <i32, granularity=8> -> i32
// CHECK-NEXT:     return
// CHECK-NEXT:   }


func.func @extract_value(%M_sym: !ktdp.runtime_arg<index, granularity=3, upperbound=300>) {
  %val = ktdp.runtime_arg_extract value from %M_sym : !ktdp.runtime_arg<index, granularity=3, upperbound=300> -> index
  return
}

func.func @extract_granularity(%M_sym: !ktdp.runtime_arg<index, granularity=3, upperbound=300>) {
  %gran = ktdp.runtime_arg_extract granularity from %M_sym : !ktdp.runtime_arg<index, granularity=3, upperbound=300> -> index
  return
}

func.func @extract_upperbound(%M_sym: !ktdp.runtime_arg<index, granularity=3, upperbound=300>) {
  %ub = ktdp.runtime_arg_extract upperbound from %M_sym : !ktdp.runtime_arg<index, granularity=3, upperbound=300> -> index
  return
}

func.func @extract_all(%M_sym: !ktdp.runtime_arg<index, granularity=3, upperbound=300>) {
  %val = ktdp.runtime_arg_extract value from %M_sym : !ktdp.runtime_arg<index, granularity=3, upperbound=300> -> index
  %gran = ktdp.runtime_arg_extract granularity from %M_sym : !ktdp.runtime_arg<index, granularity=3, upperbound=300> -> index
  %ub = ktdp.runtime_arg_extract upperbound from %M_sym : !ktdp.runtime_arg<index, granularity=3, upperbound=300> -> index
  return
}

func.func @extract_value_only_granularity(%N_sym: !ktdp.runtime_arg<index, granularity=4>) {
  %val = ktdp.runtime_arg_extract value from %N_sym : !ktdp.runtime_arg<index, granularity=4> -> index
  %gran = ktdp.runtime_arg_extract granularity from %N_sym : !ktdp.runtime_arg<index, granularity=4> -> index
  return
}

func.func @extract_i32_value(%size: !ktdp.runtime_arg<i32, granularity=8>) {
  %val = ktdp.runtime_arg_extract value from %size : !ktdp.runtime_arg<i32, granularity=8> -> i32
  return
}
