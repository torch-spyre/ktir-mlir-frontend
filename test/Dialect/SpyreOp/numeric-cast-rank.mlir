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

// RUN: ktir-opt %s --split-input-file --verify-diagnostics

// -----

func.func @roundtrip_rank3(%input: tensor<3x2x64x!spyreop.df16>) -> tensor<3x2x64x!spyreop.df16> {
  %wide = spyreop.numeric_cast %input {kind = "dl16tofp32"} : tensor<3x2x64x!spyreop.df16> -> tensor<6x2x32xf32>
  %output = spyreop.numeric_cast %wide {kind = "fp32todl16"} : tensor<6x2x32xf32> -> tensor<3x2x64x!spyreop.df16>
  return %output : tensor<3x2x64x!spyreop.df16>
}

// -----

func.func @roundtrip_rank4(%input: tensor<2x3x2x64x!spyreop.df16>) -> tensor<2x3x2x64x!spyreop.df16> {
  %output = spyreop.numeric_cast %input {kind = "dl16tofp32"} : tensor<2x3x2x64x!spyreop.df16> -> tensor<2x6x2x32xf32>
  %narrow = spyreop.numeric_cast %output {kind = "fp32todl16"} : tensor<2x6x2x32xf32> -> tensor<2x3x2x64x!spyreop.df16>
  return %narrow : tensor<2x3x2x64x!spyreop.df16>
}

// -----

func.func @roundtrip_rank5(%input: tensor<2x3x6x2x32xf32>) -> tensor<2x3x6x2x32xf32> {
  %output = spyreop.numeric_cast %input {kind = "fp32todl16"} : tensor<2x3x6x2x32xf32> -> tensor<2x3x3x2x64x!spyreop.df16>
  %wide = spyreop.numeric_cast %output {kind = "dl16tofp32"} : tensor<2x3x3x2x64x!spyreop.df16> -> tensor<2x3x6x2x32xf32>
  return %wide : tensor<2x3x6x2x32xf32>
}

// -----

func.func @roundtrip_rank6(%input: tensor<2x3x4x3x2x64x!spyreop.df16>) -> tensor<2x3x4x3x2x64x!spyreop.df16> {
  %wide = spyreop.numeric_cast %input {kind = "dl16tofp32"} : tensor<2x3x4x3x2x64x!spyreop.df16> -> tensor<2x3x4x6x2x32xf32>
  %output = spyreop.numeric_cast %wide {kind = "fp32todl16"} : tensor<2x3x4x6x2x32xf32> -> tensor<2x3x4x3x2x64x!spyreop.df16>
  return %output : tensor<2x3x4x3x2x64x!spyreop.df16>
}

// -----

func.func @cast_rank(%input: tensor<2x3x2x64x!spyreop.df16>) -> tensor<4x3x2x32xf32> {
  // expected-error@+1 {{requires tensor<Nx64x!spyreop.df16>}}
  %output = spyreop.numeric_cast %input {kind = "dl16tofp32"} : tensor<2x3x2x64x!spyreop.df16> -> tensor<4x3x2x32xf32>
  return %output : tensor<4x3x2x32xf32>
}

// -----

func.func @cast_rank(%input: tensor<2x3x2x64x!spyreop.df16>) -> tensor<3x6x2x32xf32> {
  // expected-error@+1 {{requires tensor<Nx64x!spyreop.df16>}}
  %output = spyreop.numeric_cast %input {kind = "dl16tofp32"} : tensor<2x3x2x64x!spyreop.df16> -> tensor<3x6x2x32xf32>
  return %output : tensor<3x6x2x32xf32>
}

// -----

func.func @cast_rank(%input: tensor<2x3x2x64x!spyreop.df16>) -> tensor<2x6x3x32xf32> {
  // expected-error@+1 {{requires tensor<Nx64x!spyreop.df16>}}
  %output = spyreop.numeric_cast %input {kind = "dl16tofp32"} : tensor<2x3x2x64x!spyreop.df16> -> tensor<2x6x3x32xf32>
  return %output : tensor<2x6x3x32xf32>
}

// -----

func.func @cast_rank(%input: tensor<2x3x2x64x!spyreop.df16>) -> tensor<6x2x32xf32> {
  // expected-error@+1 {{requires tensor<Nx64x!spyreop.df16>}}
  %output = spyreop.numeric_cast %input {kind = "dl16tofp32"} : tensor<2x3x2x64x!spyreop.df16> -> tensor<6x2x32xf32>
  return %output : tensor<6x2x32xf32>
}

// -----

func.func @cast_rank(%input: tensor<0x3x2x64x!spyreop.df16>) -> tensor<0x6x2x32xf32> {
  // expected-error@+1 {{requires tensor<Nx64x!spyreop.df16>}}
  %output = spyreop.numeric_cast %input {kind = "dl16tofp32"} : tensor<0x3x2x64x!spyreop.df16> -> tensor<0x6x2x32xf32>
  return %output : tensor<0x6x2x32xf32>
}

// -----

func.func @cast_rank(%input: tensor<?x3x2x64x!spyreop.df16>) -> tensor<?x6x2x32xf32> {
  // expected-error@+1 {{requires tensor<Nx64x!spyreop.df16>}}
  %output = spyreop.numeric_cast %input {kind = "dl16tofp32"} : tensor<?x3x2x64x!spyreop.df16> -> tensor<?x6x2x32xf32>
  return %output : tensor<?x6x2x32xf32>
}

// -----

func.func @wrong_packet_axis_rank3(%input: tensor<3x2x64x!spyreop.df16>) -> tensor<3x4x32xf32> {
  // expected-error@+1 {{requires tensor<Nx64x!spyreop.df16>}}
  %output = spyreop.numeric_cast %input {kind = "dl16tofp32"} : tensor<3x2x64x!spyreop.df16> -> tensor<3x4x32xf32>
  return %output : tensor<3x4x32xf32>
}

// -----

func.func @odd_packet_count_rank4(%input: tensor<2x5x2x32xf32>) -> tensor<2x2x2x64x!spyreop.df16> {
  // expected-error@+1 {{requires tensor<Nx64x!spyreop.df16>}}
  %output = spyreop.numeric_cast %input {kind = "fp32todl16"} : tensor<2x5x2x32xf32> -> tensor<2x2x2x64x!spyreop.df16>
  return %output : tensor<2x2x2x64x!spyreop.df16>
}
