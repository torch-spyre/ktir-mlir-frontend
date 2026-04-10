// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s"

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>


// CHECK:   func.func @add(%[[VAL_0:.*]]: !ktdp.runtime_arg<index, granularity=32, upperbound=300>, %[[VAL_1:.*]]: !ktdp.runtime_arg<index>) {
// CHECK-NEXT:     %[[VAL_2:.*]] = ktdp.runtime_arg_extract value from %[[VAL_0]] : <index, granularity=32, upperbound=300> -> index
// CHECK-NEXT:     %[[VAL_3:.*]] = ktdp.runtime_arg_extract value from %[[VAL_1]] : <index> -> index
// CHECK-NEXT:     %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[VAL_6:.*]] = arith.constant 12288 : index
// CHECK-NEXT:     %[[VAL_7:.*]] = arith.constant 18432 : index
// CHECK-NEXT:     %[[VAL_8:.*]] = ktdp.construct_memory_view %[[VAL_3]], sizes: {{\[}}%[[VAL_2]], 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<?x64xf16>
// CHECK-NEXT:     %[[VAL_9:.*]] = ktdp.construct_memory_view %[[VAL_6]], sizes: {{\[}}%[[VAL_2]], 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<?x64xf16>
// CHECK-NEXT:     %[[VAL_10:.*]] = ktdp.construct_memory_view %[[VAL_7]], sizes: {{\[}}%[[VAL_2]], 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<?x64xf16>
// CHECK-NEXT:     %[[VAL_11:.*]] = arith.constant 32 : index
// CHECK-NEXT:     %[[VAL_12:.*]] = arith.divsi %[[VAL_2]], %[[VAL_11]] : index
// CHECK-NEXT:     %[[VAL_13:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:     %[[VAL_14:.*]] = arith.muli %[[VAL_13]], %[[VAL_12]] : index
// CHECK-NEXT:     scf.for %[[VAL_15:.*]] = %[[VAL_4]] to %[[VAL_12]] step %[[VAL_5]] {
// CHECK-NEXT:       %[[VAL_16:.*]] = ktdp.construct_access_tile %[[VAL_8]]{{\[}}%[[VAL_14]] + %[[VAL_15]], %[[VAL_4]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<?x64xf16> -> !ktdp.access_tile<1x64xindex>
// CHECK-NEXT:       %[[VAL_17:.*]] = ktdp.construct_access_tile %[[VAL_9]]{{\[}}%[[VAL_14]] + %[[VAL_15]], %[[VAL_4]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<?x64xf16> -> !ktdp.access_tile<1x64xindex>
// CHECK-NEXT:       %[[VAL_18:.*]] = ktdp.load %[[VAL_16]] : <1x64xindex> -> tensor<1x64xf16>
// CHECK-NEXT:       %[[VAL_19:.*]] = ktdp.load %[[VAL_17]] : <1x64xindex> -> tensor<1x64xf16>
// CHECK-NEXT:       %[[VAL_20:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:       %[[VAL_21:.*]] = linalg.add ins(%[[VAL_18]], %[[VAL_19]] : tensor<1x64xf16>, tensor<1x64xf16>) outs(%[[VAL_20]] : tensor<1x64xf16>) -> tensor<1x64xf16>
// CHECK-NEXT:       %[[VAL_22:.*]] = ktdp.construct_access_tile %[[VAL_10]]{{\[}}%[[VAL_14]] + %[[VAL_15]], %[[VAL_4]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<?x64xf16> -> !ktdp.access_tile<1x64xindex>
// CHECK-NEXT:       ktdp.store %[[VAL_21]], %[[VAL_22]] : tensor<1x64xf16>, <1x64xindex>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }




#identity = affine_map<(d0, d1) -> (d0, d1)>

// An example of two tensors of sizes M(=96)x64 allocated on HBM.
// Each compute tile works at a granularity of (M/32)x64 with total number of compute tiles being 32 matching with number of cores.
module {
  func.func @add(%M_symb : !ktdp.runtime_arg<index, granularity=32, upperbound=300>,
                %A_start_address_symb : !ktdp.runtime_arg<index>) {

    %M = ktdp.runtime_arg_extract value from %M_symb: !ktdp.runtime_arg<index, granularity=32, upperbound=300> -> index
    %A_start_address = ktdp.runtime_arg_extract value from %A_start_address_symb : !ktdp.runtime_arg<index> -> index

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %B_start_address = arith.constant 12288 : index
    %C_start_address = arith.constant 18432 : index


    // Construct a memory view of A from a given address
    // s0 in coordinate set maps to %M in sizes.
    %A_view = ktdp.construct_memory_view %A_start_address, sizes: [%M, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<?x64xf16>

    // Construct a memory view of B from a given address
    %B_view = ktdp.construct_memory_view %B_start_address, sizes: [%M, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<?x64xf16>

    // Construct a memory view of C from a given address
    %C_view = ktdp.construct_memory_view %C_start_address, sizes: [%M, 64], strides: [64, 1]{
        coordinate_set = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<?x64xf16>

    // Looping over tile size with each iteration working over 1x64 fp16
    %total_tiles = arith.constant 32 : index

    // Assumption: M is divisible by total compute tiles
    %tile_size = arith.divsi %M, %total_tiles : index //M/32

    %id = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index

    scf.for %i = %c0 to %tile_size step %c1 { // loop of M/32.

        // Construct an access tile from the memory view of A
        %A_access_tile = ktdp.construct_access_tile %A_view[%start_row + %i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = #identity
        } : memref<?x64xf16> -> !ktdp.access_tile<1x64xindex>

        // Construct an access tile from the memory view of B
        %B_access_tile = ktdp.construct_access_tile %B_view[%start_row + %i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = #identity
        } : memref<?x64xf16> -> !ktdp.access_tile<1x64xindex>

        // Load data from the corresponding access tile
        %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

        %B_data_tile = ktdp.load %B_access_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

        // Perform add operation on the data tiles.
        %C_data_tile = tensor.empty() : tensor<1x64xf16>
        %C_result = linalg.add ins(%A_data_tile, %B_data_tile : tensor<1x64xf16>, tensor<1x64xf16>)
                    outs(%C_data_tile: tensor<1x64xf16>) -> tensor<1x64xf16>

        // Construct an access tile from the memory view of C
        %C_access_tile = ktdp.construct_access_tile %C_view[%start_row + %i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = #identity
        } : memref<?x64xf16> -> !ktdp.access_tile<1x64xindex>

        // Store data into the access tile.
        ktdp.store %C_result, %C_access_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    }

    return
  }
}