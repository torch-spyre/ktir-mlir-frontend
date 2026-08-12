// RUN: ktir-opt %s 

func.func @addi32toi32(%arg0: i32, %arg1: i32) -> i32 {
  %0 = spyreop.addi32toi32 %arg0, %arg1
  return %0 : i32
}

func.func @addi64toi64(%arg0: i64, %arg1: i64) -> i64 {
  %0 = spyreop.addi64toi64 %arg0, %arg1
  return %0 : i64
}

func.func @exp(%arg0: f32) -> f32 {
  %0 = spyreop.exp %arg0 : f32
  return %0 : f32
}

func.func @gelu(%arg0: !spyreop.df16) -> !spyreop.df16 {
  %0 = spyreop.gelu %arg0 : !spyreop.df16
  return %0 : !spyreop.df16
}

func.func @idx32toaddr(%arg0: i32) -> i32 {
  %0 = spyreop.idx32toaddr %arg0 base 0xDEADBEEF stride 4
  return %0 : i32
}

func.func @layernormscale(%arg0: !spyreop.df16) -> !spyreop.df16 {
  %0 = spyreop.layernormscale %arg0 : !spyreop.df16
  return %0 : !spyreop.df16
}

func.func @muli32toi32(%arg0: i32, %arg1: i32) -> i32 {
  %0 = spyreop.muli32toi32 %arg0, %arg1
  return %0 : i32
}

func.func @realdiv(%arg0: !spyreop.df16, %arg1: !spyreop.df16) -> !spyreop.df16 {
  %0 = spyreop.realdiv %arg0, %arg1 : !spyreop.df16
  return %0 : !spyreop.df16
}

func.func @reciprocal(%arg0: !spyreop.df16) -> !spyreop.df16 {
  %0 = spyreop.reciprocal %arg0 : !spyreop.df16
  return %0 : !spyreop.df16
}

func.func @sigmoid(%arg0: !spyreop.df16) -> !spyreop.df16 {
  %0 = spyreop.sigmoid %arg0 : !spyreop.df16
  return %0 : !spyreop.df16
}

func.func @softplus(%arg0: !spyreop.df16) -> !spyreop.df16 {
  %0 = spyreop.softplus %arg0 beta 0.2 threshold 0.1 : !spyreop.df16
  return %0 : !spyreop.df16
}

func.func @sqrt(%arg0: !spyreop.df16) -> !spyreop.df16 {
  %0 = spyreop.sqrt %arg0 : !spyreop.df16
  return %0 : !spyreop.df16
}
