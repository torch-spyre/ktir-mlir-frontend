// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s"


// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 31 >= 0)>
// CHECK: #[[$ATTR_4:.+]] = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_5:.+]] = affine_set<(d0, d1)[s0, s1] : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + s1 - 1 >= 0)>


// CHECK:   func.func @test_affine_expr_with_transpose(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index) {
// CHECK-NEXT:     %[[VAL_2:.*]] = arith.constant 8000 : index
// CHECK-NEXT:     %[[VAL_3:.*]] = ktdp.construct_memory_view %[[VAL_2]], sizes: [32, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_2]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<32x64xf16>
// CHECK-NEXT:     %[[VAL_4:.*]] = ktdp.construct_access_tile %[[VAL_3]]{{\[}}%[[VAL_0]] * 2, %[[VAL_1]] + 4] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_3]]} : memref<32x64xf16> -> !ktdp.access_tile<32x16xindex>
// CHECK-NEXT:     %[[VAL_5:.*]] = ktdp.construct_access_tile %[[VAL_3]]{{\[}}%[[VAL_0]] * 2, %[[VAL_1]] + 4] {access_tile_order = #[[$ATTR_1]], access_tile_set = #[[$ATTR_3]]} : memref<32x64xf16> -> !ktdp.access_tile<16x32xindex>
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func @test_with_single_symbol(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index) {
// CHECK-NEXT:     %[[VAL_3:.*]] = arith.constant 8000 : index
// CHECK-NEXT:     %[[VAL_4:.*]] = ktdp.construct_memory_view %[[VAL_3]], sizes: [32, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_2]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<32x64xf16>
// CHECK-NEXT:     %[[VAL_5:.*]] = ktdp.construct_access_tile %[[VAL_4]]{{\[}}%[[VAL_0]], %[[VAL_1]]] symbols(%[[VAL_2]]) {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_4]]} : memref<32x64xf16> -> !ktdp.access_tile<?x64xindex>
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func @test_with_two_symbols(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: index) {
// CHECK-NEXT:     %[[VAL_4:.*]] = arith.constant 8000 : index
// CHECK-NEXT:     %[[VAL_5:.*]] = ktdp.construct_memory_view %[[VAL_4]], sizes: [32, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_2]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<32x64xf16>
// CHECK-NEXT:     %[[VAL_6:.*]] = ktdp.construct_access_tile %[[VAL_5]]{{\[}}%[[VAL_0]], %[[VAL_1]]] symbols(%[[VAL_2]], %[[VAL_3]]) {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_5]]} : memref<32x64xf16> -> !ktdp.access_tile<?x?xindex>
// CHECK-NEXT:     return
// CHECK-NEXT:   }




#coord_set_A = affine_set<(d0, d1) : (d0 >= 0, -d0 + 15 >= 0,
                                      d1 >= 0, -d1 + 31 >= 0)>

#order_B = affine_map<(d0, d1) -> (d1, d0)>

// Test: Verify round-trip with affine expressions and transpose order
func.func @test_affine_expr_with_transpose(%i: index, %j: index) {
    %addr = arith.constant 8000 : index
     
    %view = ktdp.construct_memory_view %addr,
        sizes: [32, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0,
                                                 d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x64xf16>
    
    %tile_A = ktdp.construct_access_tile %view[%i * 2, %j + 4] {
        access_tile_set = #coord_set_A,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x64xf16> -> !ktdp.access_tile<32x16xindex>

    %tile_B = ktdp.construct_access_tile %view[%i * 2, %j + 4] {
        access_tile_set = #coord_set_A,
        access_tile_order = #order_B
    } : memref<32x64xf16> -> !ktdp.access_tile<16x32xindex>


    return
}

// Test: Access tile with single symbol bound to runtime value
func.func @test_with_single_symbol(%i: index, %j: index, %tile_size: index) {
    %addr = arith.constant 8000 : index
     
    %view = ktdp.construct_memory_view %addr,
        sizes: [32, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0,
                                                 d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x64xf16>
    
    // Access tile with symbolic size: 0 <= d0 < tile_size, 0 <= d1 < 64
    %tile = ktdp.construct_access_tile %view[%i, %j] symbols(%tile_size) {
        access_tile_set = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0,
                                                       d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x64xf16> -> !ktdp.access_tile<?x64xindex>

    return
}

// Test: Access tile with two symbols for dynamic tile shape
func.func @test_with_two_symbols(%i: index, %j: index, %rows: index, %cols: index) {
    %addr = arith.constant 8000 : index
     
    %view = ktdp.construct_memory_view %addr,
        sizes: [32, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0,
                                                 d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x64xf16>
    
    // Access tile with dynamic shape: 0 <= d0 < rows, 0 <= d1 < cols
    %tile = ktdp.construct_access_tile %view[%i, %j] symbols(%rows, %cols) {
        access_tile_set = affine_set<(d0, d1)[s0, s1] : (d0 >= 0, -d0 + s0 - 1 >= 0,
                                                           d1 >= 0, -d1 + s1 - 1 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x64xf16> -> !ktdp.access_tile<?x?xindex>

    return
}
