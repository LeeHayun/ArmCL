#!/bin/bash

scons Werror=0 -j8 debug=1 neon=0 opencl=1 os=linux arch=armv7a build=native examples=0

./build_example.sh

LD_LIBRARY_PATH=build ./cl_sparse_sgemm 1_5.npy 5_5.mtx

