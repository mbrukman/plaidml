// RUN: pmlc-opt -pmlc-gpu-kernel-outlining %s 

module {
  func @eltwise_add(%arg0: memref<10x20xf32>, %arg1: memref<10x20xf32>, %arg2: memref<10x20xf32>) {
    %c10 = constant 10 : index
    %c20 = constant 20 : index
    %c1 = constant 1 : index
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c10, %arg10 = %c1, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c20, %arg13 = %c1, %arg14 = %c1) {
      %0 = load %arg1[%arg3, %arg6] : memref<10x20xf32>
      %1 = load %arg0[%arg3, %arg6] : memref<10x20xf32>
      %2 = addf %0, %1 : f32
      store %2, %arg2[%arg3, %arg6] : memref<10x20xf32>
      gpu.terminator
    }
    return
  }
}

