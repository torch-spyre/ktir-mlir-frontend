// RUN: ktir-opt %s | ktir-opt | FileCheck %s



// CHECK: #[[$ATTR_0:.+]] = affine_set<(d0) : (d0 == 0)>

// CHECK-LABEL:   func.func @future_single(
// CHECK-SAME:  %[[VAL_0:.*]]: !ktdp.tile_future<(tensor<1x64xf16>), groups = #[[$ATTR_0]]>) -> !ktdp.tile_future<(tensor<1x64xf16>), groups = #[[$ATTR_0]]> {
// CHECK-NEXT:     return %[[VAL_0]] : !ktdp.tile_future<(tensor<1x64xf16>), groups = #[[$ATTR_0]]>
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @future_multi(
// CHECK-SAME:  %[[VAL_0:.*]]: !ktdp.tile_future<(tensor<128xf32>, tensor<128xi32>), groups = #[[$ATTR_0]]>) -> !ktdp.tile_future<(tensor<128xf32>, tensor<128xi32>), groups = #[[$ATTR_0]]> {
// CHECK-NEXT:     return %[[VAL_0]] : !ktdp.tile_future<(tensor<128xf32>, tensor<128xi32>), groups = #[[$ATTR_0]]>
// CHECK-NEXT:   }


#g0 = affine_set<(g) : (g == 0)>
#any = affine_set<(g) : (g == 0)>

func.func @future_single(%arg0: !ktdp.tile_future<(tensor<1x64xf16>), groups = #g0>)
    -> !ktdp.tile_future<(tensor<1x64xf16>), groups = #g0> {
  return %arg0 : !ktdp.tile_future<(tensor<1x64xf16>), groups = #g0>
}

func.func @future_multi(%arg0: !ktdp.tile_future<(tensor<128xf32>, tensor<128xi32>), groups = #any>)
    -> !ktdp.tile_future<(tensor<128xf32>, tensor<128xi32>), groups = #any> {
  return %arg0 : !ktdp.tile_future<(tensor<128xf32>, tensor<128xi32>), groups = #any>
}
