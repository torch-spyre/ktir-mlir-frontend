// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s"

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>

// CHECK-LABEL:   func.func @basic() {
// CHECK-NEXT:     %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 1024 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = ktdp.construct_memory_view %[[VAL_1]], sizes: [32, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<32x64xf16>
// CHECK-NEXT:     %[[VAL_3:.*]] = ktdp.construct_distributed_memory_view(%[[VAL_2]], %[[VAL_2]] : memref<32x64xf16>, memref<32x64xf16>) : memref<64x64xf16>
// CHECK-NEXT:     %[[VAL_4:.*]] = ktdp.construct_access_tile %[[VAL_2]]{{\[}}%[[VAL_0]], %[[VAL_0]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_1]]} : memref<32x64xf16> -> !ktdp.access_tile<32x64xindex>
// CHECK-NEXT:     %[[VAL_5:.*]] = ktdp.load %[[VAL_4]] : <32x64xindex> -> tensor<32x64xf16>
// CHECK-NEXT:     ktdp.store %[[VAL_5]], %[[VAL_4]] : tensor<32x64xf16>, <32x64xindex>
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// Round-tripping dummy test


#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @basic() {
    %c0 = arith.constant 0 : index
    %A_start_address = arith.constant 1024 : index

    // Get a memory view of the allocated space
    %A_view = ktdp.construct_memory_view %A_start_address, sizes: [32, 64], strides: [64, 1] {
        coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x64xf16>

    %A_tmp_view = ktdp.construct_distributed_memory_view (%A_view, %A_view : memref<32x64xf16>, memref<32x64xf16>) : memref<64x64xf16>

    // Get an access tile set from the tensor view
    %A_access_tile = ktdp.construct_access_tile %A_view[%c0, %c0] {
        access_tile_set = #set, access_tile_order = #identity
    } : memref<32x64xf16> -> !ktdp.access_tile<32x64xindex>

    // Load data from the corresponding access tile
    %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<32x64xindex> -> tensor<32x64xf16>

    // Store data into the access tile.
    ktdp.store %A_data_tile, %A_access_tile : tensor<32x64xf16>, !ktdp.access_tile<32x64xindex>

    return
  }
}
