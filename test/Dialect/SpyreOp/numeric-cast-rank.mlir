//===----------------------------------------------------------------------===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

// RUN: ktir-opt %s | ktir-opt | FileCheck %s
// Ranks belong to the generic tensors; the cast always operates on scalars.
// DL16 iteration coordinates map to Staggered FP32 coordinates on the output
// of widening and the input of narrowing. The intermediate tensor is staggered.

// CHECK-DAG: #[[I2:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[S2:.*]] = affine_map<(d0, d1) -> (d0 * 2 + (d1 mod 8) floordiv 4, (d1 floordiv 8) * 4 + d1 mod 4)>
// CHECK-DAG: #[[I3:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[S3:.*]] = affine_map<(d0, d1, d2) -> (d0 * 2 + (d2 mod 8) floordiv 4, d1, (d2 floordiv 8) * 4 + d2 mod 4)>
// CHECK-DAG: #[[I4:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[S4:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1 * 2 + (d3 mod 8) floordiv 4, d2, (d3 floordiv 8) * 4 + d3 mod 4)>
// CHECK-DAG: #[[I5:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[S5:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2 * 2 + (d4 mod 8) floordiv 4, d3, (d4 floordiv 8) * 4 + d4 mod 4)>
// CHECK-DAG: #[[I6:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-DAG: #[[S6:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3 * 2 + (d5 mod 8) floordiv 4, d4, (d5 floordiv 8) * 4 + d5 mod 4)>

// CHECK-LABEL: func.func @roundtrip_rank2
// CHECK: linalg.generic {indexing_maps = [#[[I2]], #[[S2]]]
// CHECK: spyreop.numeric_cast {{.*}}kind = "dl16tofp32"} : f16 -> f32
// CHECK: linalg.generic {indexing_maps = [#[[S2]], #[[I2]]]
// CHECK: spyreop.numeric_cast {{.*}}kind = "fp32todl16"} : f32 -> f16
func.func @roundtrip_rank2(%input: tensor<3x64xf16>) -> tensor<3x64xf16> {
  %cast_init_1 = tensor.empty() : tensor<6x32xf32>
  %staggered = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0 * 2 + (d1 mod 8) floordiv 4, (d1 floordiv 8) * 4 + d1 mod 4)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<3x64xf16>) outs(%cast_init_1 : tensor<6x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<6x32xf32>
  %cast_init_2 = tensor.empty() : tensor<3x64xf16>
  %output = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0 * 2 + (d1 mod 8) floordiv 4, (d1 floordiv 8) * 4 + d1 mod 4)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%staggered : tensor<6x32xf32>) outs(%cast_init_2 : tensor<3x64xf16>) {
  ^bb0(%value: f32, %unused: f16):
    %cast = spyreop.numeric_cast %value {kind = "fp32todl16"} : f32 -> f16
    linalg.yield %cast : f16
  } -> tensor<3x64xf16>
  return %output : tensor<3x64xf16>
}

// CHECK-LABEL: func.func @roundtrip_rank3
// CHECK: linalg.generic {indexing_maps = [#[[I3]], #[[S3]]]
// CHECK: spyreop.numeric_cast {{.*}}kind = "dl16tofp32"} : f16 -> f32
// CHECK: linalg.generic {indexing_maps = [#[[S3]], #[[I3]]]
// CHECK: spyreop.numeric_cast {{.*}}kind = "fp32todl16"} : f32 -> f16
func.func @roundtrip_rank3(%input: tensor<3x2x64xf16>) -> tensor<3x2x64xf16> {
  %cast_init_1 = tensor.empty() : tensor<6x2x32xf32>
  %staggered = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0 * 2 + (d2 mod 8) floordiv 4, d1, (d2 floordiv 8) * 4 + d2 mod 4)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<3x2x64xf16>) outs(%cast_init_1 : tensor<6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<6x2x32xf32>
  %cast_init_2 = tensor.empty() : tensor<3x2x64xf16>
  %output = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0 * 2 + (d2 mod 8) floordiv 4, d1, (d2 floordiv 8) * 4 + d2 mod 4)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%staggered : tensor<6x2x32xf32>) outs(%cast_init_2 : tensor<3x2x64xf16>) {
  ^bb0(%value: f32, %unused: f16):
    %cast = spyreop.numeric_cast %value {kind = "fp32todl16"} : f32 -> f16
    linalg.yield %cast : f16
  } -> tensor<3x2x64xf16>
  return %output : tensor<3x2x64xf16>
}

// CHECK-LABEL: func.func @roundtrip_rank4
// CHECK: linalg.generic {indexing_maps = [#[[I4]], #[[S4]]]
// CHECK: spyreop.numeric_cast {{.*}}kind = "dl16tofp32"} : f16 -> f32
// CHECK: linalg.generic {indexing_maps = [#[[S4]], #[[I4]]]
// CHECK: spyreop.numeric_cast {{.*}}kind = "fp32todl16"} : f32 -> f16
func.func @roundtrip_rank4(%input: tensor<2x3x2x64xf16>) -> tensor<2x3x2x64xf16> {
  %cast_init_1 = tensor.empty() : tensor<2x6x2x32xf32>
  %staggered = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1 * 2 + (d3 mod 8) floordiv 4, d2, (d3 floordiv 8) * 4 + d3 mod 4)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<2x3x2x64xf16>) outs(%cast_init_1 : tensor<2x6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<2x6x2x32xf32>
  %cast_init_2 = tensor.empty() : tensor<2x3x2x64xf16>
  %output = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1 * 2 + (d3 mod 8) floordiv 4, d2, (d3 floordiv 8) * 4 + d3 mod 4)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%staggered : tensor<2x6x2x32xf32>) outs(%cast_init_2 : tensor<2x3x2x64xf16>) {
  ^bb0(%value: f32, %unused: f16):
    %cast = spyreop.numeric_cast %value {kind = "fp32todl16"} : f32 -> f16
    linalg.yield %cast : f16
  } -> tensor<2x3x2x64xf16>
  return %output : tensor<2x3x2x64xf16>
}

// CHECK-LABEL: func.func @roundtrip_rank5
// CHECK: linalg.generic {indexing_maps = [#[[I5]], #[[S5]]]
// CHECK: spyreop.numeric_cast {{.*}}kind = "dl16tofp32"} : f16 -> f32
// CHECK: linalg.generic {indexing_maps = [#[[S5]], #[[I5]]]
// CHECK: spyreop.numeric_cast {{.*}}kind = "fp32todl16"} : f32 -> f16
func.func @roundtrip_rank5(%input: tensor<2x2x3x2x64xf16>) -> tensor<2x2x3x2x64xf16> {
  %cast_init_1 = tensor.empty() : tensor<2x2x6x2x32xf32>
  %staggered = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2 * 2 + (d4 mod 8) floordiv 4, d3, (d4 floordiv 8) * 4 + d4 mod 4)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<2x2x3x2x64xf16>) outs(%cast_init_1 : tensor<2x2x6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<2x2x6x2x32xf32>
  %cast_init_2 = tensor.empty() : tensor<2x2x3x2x64xf16>
  %output = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2 * 2 + (d4 mod 8) floordiv 4, d3, (d4 floordiv 8) * 4 + d4 mod 4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%staggered : tensor<2x2x6x2x32xf32>) outs(%cast_init_2 : tensor<2x2x3x2x64xf16>) {
  ^bb0(%value: f32, %unused: f16):
    %cast = spyreop.numeric_cast %value {kind = "fp32todl16"} : f32 -> f16
    linalg.yield %cast : f16
  } -> tensor<2x2x3x2x64xf16>
  return %output : tensor<2x2x3x2x64xf16>
}

// CHECK-LABEL: func.func @roundtrip_rank6
// CHECK: linalg.generic {indexing_maps = [#[[I6]], #[[S6]]]
// CHECK: spyreop.numeric_cast {{.*}}kind = "dl16tofp32"} : f16 -> f32
// CHECK: linalg.generic {indexing_maps = [#[[S6]], #[[I6]]]
// CHECK: spyreop.numeric_cast {{.*}}kind = "fp32todl16"} : f32 -> f16
func.func @roundtrip_rank6(%input: tensor<2x2x2x3x2x64xf16>) -> tensor<2x2x2x3x2x64xf16> {
  %cast_init_1 = tensor.empty() : tensor<2x2x2x6x2x32xf32>
  %staggered = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>,
                     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3 * 2 + (d5 mod 8) floordiv 4, d4, (d5 floordiv 8) * 4 + d5 mod 4)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<2x2x2x3x2x64xf16>) outs(%cast_init_1 : tensor<2x2x2x6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<2x2x2x6x2x32xf32>
  %cast_init_2 = tensor.empty() : tensor<2x2x2x3x2x64xf16>
  %output = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3 * 2 + (d5 mod 8) floordiv 4, d4, (d5 floordiv 8) * 4 + d5 mod 4)>,
                     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%staggered : tensor<2x2x2x6x2x32xf32>) outs(%cast_init_2 : tensor<2x2x2x3x2x64xf16>) {
  ^bb0(%value: f32, %unused: f16):
    %cast = spyreop.numeric_cast %value {kind = "fp32todl16"} : f32 -> f16
    linalg.yield %cast : f16
  } -> tensor<2x2x2x3x2x64xf16>
  return %output : tensor<2x2x2x3x2x64xf16>
}
