# Copyright 2020 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//vendor/mlir:tblgen.bzl", "COPTS")

plaidml_cc_library(
    name = "gpu_kernel_outlining",
    srcs = [
        "kernel_outlining.cc",
        "kernel_outlining.h",
    ],
    copts = COPTS,
    tags = ["llvm"],
    deps = [
        "//pmlc/util",
        "@llvm-project//mlir:GPUTransforms",
    ],
    alwayslink = 1,
)

