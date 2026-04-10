// RUN: ktir-opt -allow-unregistered-dialect "%s" | ktir-opt -allow-unregistered-dialect | FileCheck "%s"

// CHECK-LABEL:   func.func @comprehensive_tile_test() {
// CHECK-NEXT:     %[[VAL_0:.*]] = "test.op"() : () -> !ktdp.access_tile<1x64xindex>
// CHECK-NEXT:     %[[VAL_1:.*]] = "test.op"() : () -> !ktdp.access_tile<2x32xindex>
// CHECK-NEXT:     %[[VAL_2:.*]] = "test.op"() : () -> !ktdp.access_tile<4x16xindex>
// CHECK-NEXT:     %[[VAL_3:.*]] = "test.op"() : () -> !ktdp.access_tile<8x8xindex>
// CHECK-NEXT:     %[[VAL_4:.*]] = "test.op"() : () -> !ktdp.access_tile<?x64xindex>
// CHECK-NEXT:     %[[VAL_5:.*]] = "test.op"() : () -> !ktdp.access_tile<1x?xindex>
// CHECK-NEXT:     %[[VAL_6:.*]] = "test.op"() : () -> !ktdp.access_tile<?x?xindex>
// CHECK-NEXT:     %[[VAL_7:.*]] = "test.op"() : () -> !ktdp.access_tile<index>
// CHECK-NEXT:     %[[VAL_8:.*]] = "test.op"() : () -> !ktdp.access_tile<2x3x4xindex>
// CHECK-NEXT:     %[[VAL_9:.*]] = "test.op"() : () -> !ktdp.access_tile<1x2x3x4xindex>
// CHECK-NEXT:     return
// CHECK-NEXT:   }


module {
  func.func @comprehensive_tile_test() {
    // Static shaped tiles with index element type
    %0 = "test.op"() : () -> !ktdp.access_tile<1x64xindex>
    %1 = "test.op"() : () -> !ktdp.access_tile<2x32xindex>
    %2 = "test.op"() : () -> !ktdp.access_tile<4x16xindex>
    %3 = "test.op"() : () -> !ktdp.access_tile<8x8xindex>
    
    // Dynamic shaped tiles
    %4 = "test.op"() : () -> !ktdp.access_tile<?x64xindex>
    %5 = "test.op"() : () -> !ktdp.access_tile<1x?xindex>
    %6 = "test.op"() : () -> !ktdp.access_tile<?x?xindex>
    
    // Scalar tile (0-D)
    %7 = "test.op"() : () -> !ktdp.access_tile<index>
    
    // Multi-dimensional tiles
    %8 = "test.op"() : () -> !ktdp.access_tile<2x3x4xindex>
    %9 = "test.op"() : () -> !ktdp.access_tile<1x2x3x4xindex>
    
    return
  }
}