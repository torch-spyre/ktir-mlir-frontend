// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s"
// Round-tripping dummy test

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>


// CHECK-LABEL:   func.func @add() {
// CHECK-NEXT:     %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 3 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = arith.constant 1024 : index
// CHECK-NEXT:     %[[VAL_3:.*]] = arith.constant 12288 : index
// CHECK-NEXT:     %[[VAL_4:.*]] = arith.constant 18432 : index
// CHECK-NEXT:     %[[VAL_5:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:     %[[VAL_6:.*]] = arith.muli %[[VAL_5]], %[[VAL_1]] : index
// CHECK-NEXT:     %[[VAL_7:.*]] = ktdp.construct_memory_view %[[VAL_2]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16>
// CHECK-NEXT:     %[[VAL_8:.*]] = ktdp.construct_access_tile %[[VAL_7]]{{\[}}%[[VAL_6]], %[[VAL_0]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>
// CHECK-NEXT:     %[[VAL_9:.*]] = ktdp.construct_memory_view %[[VAL_3]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16>
// CHECK-NEXT:     %[[VAL_10:.*]] = ktdp.construct_access_tile %[[VAL_9]]{{\[}}%[[VAL_6]], %[[VAL_0]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>
// CHECK-NEXT:     %[[VAL_11:.*]] = ktdp.load %[[VAL_8]] : <3x64xindex> -> tensor<3x64xf16>
// CHECK-NEXT:     %[[VAL_12:.*]] = ktdp.load %[[VAL_10]] : <3x64xindex> -> tensor<3x64xf16>
// CHECK-NEXT:     %[[VAL_13:.*]] = tensor.empty() : tensor<3x64xf16>
// CHECK-NEXT:     %[[VAL_14:.*]] = linalg.add ins(%[[VAL_11]], %[[VAL_12]] : tensor<3x64xf16>, tensor<3x64xf16>) outs(%[[VAL_13]] : tensor<3x64xf16>) -> tensor<3x64xf16>
// CHECK-NEXT:     %[[VAL_15:.*]] = tensor.extract_slice %[[VAL_14]][0, 0] [1, 64] [1, 1] : tensor<3x64xf16> to tensor<1x64xf16>
// CHECK-NEXT:     %[[VAL_16:.*]] = ktdp.construct_memory_view %[[VAL_4]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16>
// CHECK-NEXT:     %[[VAL_17:.*]] = ktdp.construct_access_tile %[[VAL_16]]{{\[}}%[[VAL_6]], %[[VAL_0]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_3]]} : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>
// CHECK-NEXT:     ktdp.store %[[VAL_15]], %[[VAL_17]] : tensor<1x64xf16>, <1x64xindex>
// CHECK-NEXT:     return
// CHECK-NEXT:   }


// An example to demonstrate the usage of tensor-extract operation.
module {
  func.func @add() {
    %c0 = arith.constant 0 : index
    %tile_size = arith.constant 3 : index
    %A_start_address = arith.constant 1024 : index
    %B_start_address = arith.constant 12288 : index
    %C_start_address = arith.constant 18432 : index

    %id = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index

    // Construct a memory view of A from a given address
    %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    // Construct an access tile set from the memory view of A
    %A_access_tile = ktdp.construct_access_tile %A_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>

    // Construct a memory view of B from a given address
    %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    // Construct an access tile set from the memory view of B
    %B_access_tile = ktdp.construct_access_tile %B_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>

    // Load data from the corresponding access tile
    %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<3x64xindex> -> tensor<3x64xf16>

    %B_data_tile = ktdp.load %B_access_tile : !ktdp.access_tile<3x64xindex> -> tensor<3x64xf16>

    // Perform add operation on the data tiles.
    %C_data_tile = tensor.empty() : tensor<3x64xf16>
    %C_result = linalg.add ins(%A_data_tile, %B_data_tile : tensor<3x64xf16>, tensor<3x64xf16>)
                outs(%C_data_tile: tensor<3x64xf16>) -> tensor<3x64xf16>

    // extract 1x64 slice of C_result
    %C_data_subtile = tensor.extract_slice %C_result[0, 0] [1, 64] [1, 1]
      : tensor<3x64xf16> to tensor<1x64xf16>

    // Construct a memory view of C from a given address
    %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    // Construct an access tile set from the memory view of C
    %C_access_tile = ktdp.construct_access_tile %C_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

    // Store data into the access tile.
    ktdp.store %C_data_subtile, %C_access_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>

    return
  }
}
