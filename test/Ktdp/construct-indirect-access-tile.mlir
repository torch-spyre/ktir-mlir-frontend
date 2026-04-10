// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s"


// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<() -> ()>
// CHECK: #[[$ATTR_4:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>
// CHECK: #[[$ATTR_5:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 99 >= 0, d1 >= 0, -d1 + 15 >= 0)>
// CHECK: #[[$ATTR_6:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 15 >= 0)>
// CHECK: #[[$ATTR_7:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 31 >= 0)>
// CHECK: #[[$ATTR_8:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 999 >= 0, d1 >= 0, -d1 + 7 >= 0, d2 >= 0, -d2 + 127 >= 0)>
// CHECK: #[[$ATTR_9:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 7 >= 0, d2 >= 0, -d2 + 127 >= 0)>
// CHECK: #[[$ATTR_10:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 7 >= 0)>
// CHECK: #[[$ATTR_11:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 99 >= 0, d1 >= 0, -d1 + 99 >= 0)>
// CHECK: #[[$ATTR_12:.+]] = affine_set<() : (0 == 0)>
// CHECK: #[[$ATTR_13:.+]] = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + 15 >= 0)>


// CHECK-LABEL:   func.func @test_mixed_subscripts_identity_order() {
// CHECK-NEXT:     %[[VAL_0:.*]] = arith.constant 10000 : index
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 20000 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = ktdp.construct_memory_view %[[VAL_1]], sizes: [8], strides: [1] {coordinate_set = #[[$ATTR_4]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<8xi32>
// CHECK-NEXT:     %[[VAL_3:.*]] = ktdp.construct_memory_view %[[VAL_0]], sizes: [100, 16], strides: [16, 1] {coordinate_set = #[[$ATTR_5]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<100x16xf16>
// CHECK-NEXT:     %[[VAL_4:.*]] = ktdp.construct_indirect_access_tile intermediate_variables(%[[VAL_5:.*]], %[[VAL_6:.*]]) %[[VAL_3]][(%[[VAL_5]]), ind(%[[VAL_2]]{{\[}}%[[VAL_6]]])] {variables_space_order = #[[$ATTR_0]], variables_space_set = #[[$ATTR_6]]} : memref<100x16xf16>, memref<8xi32> -> !ktdp.access_tile<8x16xindex>
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @test_mixed_subscripts_transpose_order() {
// CHECK-NEXT:     %[[VAL_0:.*]] = arith.constant 10000 : index
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 20000 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = ktdp.construct_memory_view %[[VAL_1]], sizes: [8], strides: [1] {coordinate_set = #[[$ATTR_4]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<8xi32>
// CHECK-NEXT:     %[[VAL_3:.*]] = ktdp.construct_memory_view %[[VAL_0]], sizes: [100, 16], strides: [16, 1] {coordinate_set = #[[$ATTR_5]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<100x16xf16>
// CHECK-NEXT:     %[[VAL_4:.*]] = ktdp.construct_indirect_access_tile intermediate_variables(%[[VAL_5:.*]], %[[VAL_6:.*]]) %[[VAL_3]][ind(%[[VAL_2]]{{\[}}%[[VAL_5]]]), (%[[VAL_6]])] {variables_space_order = #[[$ATTR_1]], variables_space_set = #[[$ATTR_6]]} : memref<100x16xf16>, memref<8xi32> -> !ktdp.access_tile<16x8xindex>
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @test_complex_affine_expressions() {
// CHECK-NEXT:     %[[VAL_0:.*]] = arith.constant 10000 : index
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 20000 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = ktdp.construct_memory_view %[[VAL_1]], sizes: [4, 32], strides: [32, 1] {coordinate_set = #[[$ATTR_7]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<4x32xi32>
// CHECK-NEXT:     %[[VAL_3:.*]] = ktdp.construct_memory_view %[[VAL_0]], sizes: [1000, 8, 128], strides: [1024, 128, 1] {coordinate_set = #[[$ATTR_8]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<1000x8x128xf16>
// CHECK-NEXT:     %[[VAL_4:.*]] = ktdp.construct_indirect_access_tile intermediate_variables(%[[VAL_5:.*]], %[[VAL_6:.*]], %[[VAL_7:.*]]) %[[VAL_3]][ind(%[[VAL_2]]{{\[}}%[[VAL_5]], %[[VAL_7]] floordiv 64]), (%[[VAL_6]]), (%[[VAL_7]] mod 64)] {variables_space_order = #[[$ATTR_2]], variables_space_set = #[[$ATTR_9]]} : memref<1000x8x128xf16>, memref<4x32xi32> -> !ktdp.access_tile<4x8x128xindex>
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @test_multiple_indirect_dimensions() {
// CHECK-NEXT:     %[[VAL_0:.*]] = arith.constant 10000 : index
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 20000 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = arith.constant 30000 : index
// CHECK-NEXT:     %[[VAL_3:.*]] = ktdp.construct_memory_view %[[VAL_1]], sizes: [4, 8], strides: [8, 1] {coordinate_set = #[[$ATTR_10]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<4x8xi32>
// CHECK-NEXT:     %[[VAL_4:.*]] = ktdp.construct_memory_view %[[VAL_2]], sizes: [4, 8], strides: [8, 1] {coordinate_set = #[[$ATTR_10]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<4x8xi32>
// CHECK-NEXT:     %[[VAL_5:.*]] = ktdp.construct_memory_view %[[VAL_0]], sizes: [100, 100], strides: [100, 1] {coordinate_set = #[[$ATTR_11]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<100x100xf16>
// CHECK-NEXT:     %[[VAL_6:.*]] = ktdp.construct_indirect_access_tile intermediate_variables(%[[VAL_7:.*]], %[[VAL_8:.*]]) %[[VAL_5]][ind(%[[VAL_3]]{{\[}}%[[VAL_7]], %[[VAL_8]]]), ind(%[[VAL_4]]{{\[}}%[[VAL_7]], %[[VAL_8]]])] {variables_space_order = #[[$ATTR_0]], variables_space_set = #[[$ATTR_10]]} : memref<100x100xf16>, memref<4x8xi32>, memref<4x8xi32> -> !ktdp.access_tile<4x8xindex>
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func @test_with_captured_variables(%[[VAL_0:.*]]: index) {
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 10000 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = arith.constant 20000 : index
// CHECK-NEXT:     %[[VAL_3:.*]] = ktdp.construct_memory_view %[[VAL_2]], sizes: [4, 8], strides: [8, 1] {coordinate_set = #[[$ATTR_10]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<4x8xi32>
// CHECK-NEXT:     %[[VAL_4:.*]] = ktdp.construct_memory_view %[[VAL_1]], sizes: [100, 100], strides: [100, 1] {coordinate_set = #[[$ATTR_11]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<100x100xf16>
// CHECK-NEXT:     %[[VAL_5:.*]] = ktdp.construct_indirect_access_tile intermediate_variables(%[[VAL_6:.*]], %[[VAL_7:.*]]) %[[VAL_4]][ind(%[[VAL_3]]{{\[}}%[[VAL_0]] + %[[VAL_6]], %[[VAL_7]]]), (%[[VAL_7]])] {variables_space_order = #[[$ATTR_0]], variables_space_set = #[[$ATTR_10]]} : memref<100x100xf16>, memref<4x8xi32> -> !ktdp.access_tile<4x8xindex>
// CHECK-NEXT:     %[[VAL_8:.*]] = ktdp.construct_indirect_access_tile intermediate_variables() %[[VAL_4]][ind(%[[VAL_3]]{{\[}}%[[VAL_0]], 0]), (%[[VAL_0]])] {variables_space_order = #[[$ATTR_3]], variables_space_set = #[[$ATTR_12]]} : memref<100x100xf16>, memref<4x8xi32> -> !ktdp.access_tile<index>
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func @test_with_single_symbol(%[[VAL_0:.*]]: index) {
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 10000 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = arith.constant 20000 : index
// CHECK-NEXT:     %[[VAL_3:.*]] = ktdp.construct_memory_view %[[VAL_2]], sizes: [8], strides: [1] {coordinate_set = #[[$ATTR_4]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<8xi32>
// CHECK-NEXT:     %[[VAL_4:.*]] = ktdp.construct_memory_view %[[VAL_1]], sizes: [100, 16], strides: [16, 1] {coordinate_set = #[[$ATTR_5]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<100x16xf16>
// CHECK-NEXT:     %[[VAL_5:.*]] = ktdp.construct_indirect_access_tile intermediate_variables(%[[VAL_6:.*]], %[[VAL_7:.*]]) %[[VAL_4]][(%[[VAL_6]]), ind(%[[VAL_3]]{{\[}}%[[VAL_7]]])] symbols(%[[VAL_0]]) {variables_space_order = #[[$ATTR_0]], variables_space_set = #[[$ATTR_13]]} : memref<100x16xf16>, memref<8xi32> -> !ktdp.access_tile<?x16xindex>
// CHECK-NEXT:     return
// CHECK-NEXT:   }




// Global affine sets
#var_space_2d_8x16 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0,
                                             d1 >= 0, -d1 + 15 >= 0)>
#var_space_2d_4x8 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0,
                                            d1 >= 0, -d1 + 7 >= 0)>
#var_space_3d_4x8x128 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0,
                                                    d1 >= 0, -d1 + 7 >= 0,
                                                    d2 >= 0, -d2 + 127 >= 0)>

#coord_set_1d_8 = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>
#coord_set_2d_100x16 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 99 >= 0,
                                               d1 >= 0, -d1 + 15 >= 0)>
#coord_set_2d_4x32 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0,
                                             d1 >= 0, -d1 + 31 >= 0)>
#coord_set_2d_100x100 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 99 >= 0,
                                                d1 >= 0, -d1 + 99 >= 0)>
#coord_set_3d_1000x8x128 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 999 >= 0,
                                                       d1 >= 0, -d1 + 7 >= 0,
                                                       d2 >= 0, -d2 + 127 >= 0)>

#empty_set = affine_set<() : ()>
#empty_map = affine_map<() -> ()>

// Global affine maps
#order_2d_identity = affine_map<(d0, d1) -> (d0, d1)>
#order_2d_transpose = affine_map<(d0, d1) -> (d1, d0)>
#order_3d_identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// Test: Verify round-trip with mixed direct/indirect subscripts and identity order
func.func @test_mixed_subscripts_identity_order() {
    %addr_x = arith.constant 10000 : index
    %addr_idx = arith.constant 20000 : index
    
    %idx_view = ktdp.construct_memory_view %addr_idx,
        sizes: [8], strides: [1] {
        coordinate_set = #coord_set_1d_8,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xi32>
    
    %x_view = ktdp.construct_memory_view %addr_x,
        sizes: [100, 16], strides: [16, 1] {
        coordinate_set = #coord_set_2d_100x16,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<100x16xf16>
    
    %tile_A = ktdp.construct_indirect_access_tile
        intermediate_variables(%i, %j)
        %x_view[(%i), ind(%idx_view[%j])] {
        variables_space_set = #var_space_2d_8x16,
        variables_space_order = #order_2d_identity
    } : memref<100x16xf16>, memref<8xi32> -> !ktdp.access_tile<8x16xindex>
    
    return
}

// Test: Verify round-trip with mixed direct/indirect subscripts and transpose order
func.func @test_mixed_subscripts_transpose_order() {
    %addr_x = arith.constant 10000 : index
    %addr_idx = arith.constant 20000 : index
    
    %idx_view = ktdp.construct_memory_view %addr_idx,
        sizes: [8], strides: [1] {
        coordinate_set = #coord_set_1d_8,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xi32>
    
    %x_view = ktdp.construct_memory_view %addr_x,
        sizes: [100, 16], strides: [16, 1] {
        coordinate_set = #coord_set_2d_100x16,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<100x16xf16>
    
    %tile_B = ktdp.construct_indirect_access_tile
        intermediate_variables(%i, %j)
        %x_view[ind(%idx_view[%i]), (%j)] {
        variables_space_set = #var_space_2d_8x16,
        variables_space_order = #order_2d_transpose
    } : memref<100x16xf16>, memref<8xi32> -> !ktdp.access_tile<16x8xindex>
    
    return
}

// Test: Verify round-trip with complex affine expressions in subscripts
func.func @test_complex_affine_expressions() {
    %addr_x = arith.constant 10000 : index
    %addr_idx = arith.constant 20000 : index
    
    %idx_view = ktdp.construct_memory_view %addr_idx,
        sizes: [4, 32], strides: [32, 1] {
        coordinate_set = #coord_set_2d_4x32,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x32xi32>
    
    %x_view = ktdp.construct_memory_view %addr_x,
        sizes: [1000, 8, 128], strides: [1024, 128, 1] {
        coordinate_set = #coord_set_3d_1000x8x128,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1000x8x128xf16>
    
    %tile = ktdp.construct_indirect_access_tile
        intermediate_variables(%b, %h, %t)
        %x_view[ind(%idx_view[%b, (%t floordiv 64)]), (%h), (%t mod 64)] {
        variables_space_set = #var_space_3d_4x8x128,
        variables_space_order = #order_3d_identity
    } : memref<1000x8x128xf16>, memref<4x32xi32> -> !ktdp.access_tile<4x8x128xindex>
    
    return
}

// Test: Verify round-trip with multiple indirect dimensions
func.func @test_multiple_indirect_dimensions() {
    %addr_x = arith.constant 10000 : index
    %addr_idx1 = arith.constant 20000 : index
    %addr_idx2 = arith.constant 30000 : index
    
    %idx1_view = ktdp.construct_memory_view %addr_idx1,
        sizes: [4, 8], strides: [8, 1] {
        coordinate_set = #var_space_2d_4x8,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x8xi32>
    
    %idx2_view = ktdp.construct_memory_view %addr_idx2,
        sizes: [4, 8], strides: [8, 1] {
        coordinate_set = #var_space_2d_4x8,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x8xi32>
    
    %x_view = ktdp.construct_memory_view %addr_x,
        sizes: [100, 100], strides: [100, 1] {
        coordinate_set = #coord_set_2d_100x100,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<100x100xf16>

    %tile = ktdp.construct_indirect_access_tile
        intermediate_variables(%i, %j)
        %x_view[ind(%idx1_view[%i, %j]), ind(%idx2_view[%i, %j])] {
        variables_space_set = #var_space_2d_4x8,
        variables_space_order = #order_2d_identity
    } : memref<100x100xf16>, memref<4x8xi32>, memref<4x8xi32> -> !ktdp.access_tile<4x8xindex>
    
    return
}

// Test: Verify round-trip with captured variables (external SSA values)
func.func @test_with_captured_variables(%offset: index) {
    %addr_x = arith.constant 10000 : index
    %addr_idx = arith.constant 20000 : index
    
    %idx_view = ktdp.construct_memory_view %addr_idx,
        sizes: [4, 8], strides: [8, 1] {
        coordinate_set = #var_space_2d_4x8,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x8xi32>
    
    %x_view = ktdp.construct_memory_view %addr_x,
        sizes: [100, 100], strides: [100, 1] {
        coordinate_set = #coord_set_2d_100x100,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<100x100xf16>
    
    %tile = ktdp.construct_indirect_access_tile
        intermediate_variables(%i, %j)
        %x_view[ind(%idx_view[%i + %offset, %j]), (%j)] {
        variables_space_set = #var_space_2d_4x8,
        variables_space_order = #order_2d_identity
    } : memref<100x100xf16>, memref<4x8xi32> -> !ktdp.access_tile<4x8xindex>

    %scalar_tile = ktdp.construct_indirect_access_tile
        intermediate_variables()
        %x_view[ind(%idx_view[%offset, 0]), (%offset)] {
        variables_space_set = #empty_set,
        variables_space_order = #empty_map
    } : memref<100x100xf16>, memref<4x8xi32> -> !ktdp.access_tile<index>

    return
}
// Test: Indirect access tile with symbol for dynamic tile size
func.func @test_with_single_symbol(%tile_rows: index) {
    %addr_x = arith.constant 10000 : index
    %addr_idx = arith.constant 20000 : index
    
    %idx_view = ktdp.construct_memory_view %addr_idx,
        sizes: [8], strides: [1] {
        coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xi32>
    
    %x_view = ktdp.construct_memory_view %addr_x,
        sizes: [100, 16], strides: [16, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 99 >= 0, d1 >= 0, -d1 + 15 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<100x16xf16>
    
    // Indirect access tile with symbolic row count: 0 <= i < tile_rows, 0 <= j < 16
    %tile = ktdp.construct_indirect_access_tile
        intermediate_variables(%i, %j)
        %x_view[(%i), ind(%idx_view[%j])] symbols(%tile_rows) {
        variables_space_set = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0,
                                                          d1 >= 0, -d1 + 15 >= 0)>,
        variables_space_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<100x16xf16>, memref<8xi32> -> !ktdp.access_tile<?x16xindex>

    return
}
