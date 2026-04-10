// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s"

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>

// CHECK-LABEL:   func.func @basic_sym(
// CHECK-SAME:                         %[[VAL_0:.*]]: !ktdp.runtime_arg<index, granularity=3, upperbound=300>) {
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = ktdp.runtime_arg_extract value from %[[VAL_0]] : <index, granularity=3, upperbound=300> -> index
// CHECK-NEXT:     %[[VAL_3:.*]] = arith.constant 1024 : index
// CHECK-NEXT:     %[[VAL_4:.*]] = ktdp.construct_memory_view %[[VAL_3]], sizes: {{\[}}%[[VAL_2]], 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<?x64xf16>
// CHECK-NEXT:     %[[VAL_5:.*]] = ktdp.construct_distributed_memory_view(%[[VAL_4]], %[[VAL_4]] : memref<?x64xf16>, memref<?x64xf16>) : memref<?x64xf16>
// CHECK-NEXT:     %[[VAL_6:.*]] = ktdp.construct_access_tile %[[VAL_4]]{{\[}}%[[VAL_1]], %[[VAL_1]]] symbols(%[[VAL_2]]) {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_1]]} : memref<?x64xf16> -> !ktdp.access_tile<32x64xindex>
// CHECK-NEXT:     %[[VAL_7:.*]] = ktdp.load %[[VAL_6]] : <32x64xindex> -> tensor<32x64xf16>
// CHECK-NEXT:     ktdp.store %[[VAL_7]], %[[VAL_6]] : tensor<32x64xf16>, <32x64xindex>
// CHECK-NEXT:     return
// CHECK-NEXT:   }



// Round-tripping dummy test
#set = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0-1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @basic_sym(%M_Sym : !ktdp.runtime_arg<index, granularity=3, upperbound=300>) {
    %c0 = arith.constant 0 : index
    %M = ktdp.runtime_arg_extract value from %M_Sym : <index, granularity=3, upperbound=300> -> index
    %A_start_address = arith.constant 1024 : index

    // Get a memory view of the allocated space
    %A_view = ktdp.construct_memory_view %A_start_address, sizes: [%M, 64], strides: [64, 1] {
        coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<?x64xf16>

    %A_tmp_view = ktdp.construct_distributed_memory_view (%A_view, %A_view : memref<?x64xf16>, memref<?x64xf16>) : memref<?x64xf16>

    // Get an access tile set from the tensor view
    %A_access_tile = ktdp.construct_access_tile %A_view[%c0, %c0] symbols(%M) {
        access_tile_set = #set, access_tile_order = #identity
    } : memref<?x64xf16> -> !ktdp.access_tile<32x64xindex>

    // Load data from the corresponding access tile
    %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<32x64xindex> -> tensor<32x64xf16>

    // Store data into the access tile.
    ktdp.store %A_data_tile, %A_access_tile : tensor<32x64xf16>, !ktdp.access_tile<32x64xindex>

    return
  }
}
