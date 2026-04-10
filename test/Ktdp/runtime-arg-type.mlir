// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s"

// CHECK:   func.func @runtime_arg_basic(%[[VAL_0:.*]]: !ktdp.runtime_arg<index>) {
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func @runtime_arg_with_granularity(%[[VAL_0:.*]]: !ktdp.runtime_arg<index, granularity=3>) {
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func @runtime_arg_with_upperbound(%[[VAL_0:.*]]: !ktdp.runtime_arg<index, upperbound=300>) {
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func @runtime_arg_with_both(%[[VAL_0:.*]]: !ktdp.runtime_arg<index, granularity=3, upperbound=300>) {
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func @runtime_arg_i32(%[[VAL_0:.*]]: !ktdp.runtime_arg<i32, granularity=4, upperbound=1024>) {
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func @runtime_arg_i64(%[[VAL_0:.*]]: !ktdp.runtime_arg<i64>) {
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func @runtime_arg_f32(%[[VAL_0:.*]]: !ktdp.runtime_arg<f32, upperbound=100>) {
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func @multiple_runtime_args(%[[VAL_0:.*]]: !ktdp.runtime_arg<index, granularity=3, upperbound=300>, %[[VAL_1:.*]]: !ktdp.runtime_arg<index, granularity=3, upperbound=300>, %[[VAL_2:.*]]: !ktdp.runtime_arg<index>) {
// CHECK-NEXT:     return
// CHECK-NEXT:   }


func.func @runtime_arg_basic(%arg0: !ktdp.runtime_arg<index>) {
  return
}

func.func @runtime_arg_with_granularity(%M_Sym: !ktdp.runtime_arg<index, granularity=3>) {
  return
}

func.func @runtime_arg_with_upperbound(%N_Sym: !ktdp.runtime_arg<index, upperbound=300>) {
  return
}

func.func @runtime_arg_with_both(%M_Sym: !ktdp.runtime_arg<index, granularity=3, upperbound=300>) {
  return
}

func.func @runtime_arg_i32(%size: !ktdp.runtime_arg<i32, granularity=4, upperbound=1024>) {
  return
}

func.func @runtime_arg_i64(%count: !ktdp.runtime_arg<i64>) {
  return
}

func.func @runtime_arg_f32(%scale: !ktdp.runtime_arg<f32, upperbound=100>) {
  return
}

func.func @multiple_runtime_args(
    %M: !ktdp.runtime_arg<index, granularity=3, upperbound=300>,
    %N: !ktdp.runtime_arg<index, granularity=3, upperbound=300>,
    %K: !ktdp.runtime_arg<index>) {
  return
}
