#!/bin/bash

#g++ examples/cl_sparse_sgemm.cpp build/utils/*.o -I. -Iinclude -std=c++11 -mfpu=neon -Lbuild -larm_compute -larm_compute_core -lpthread -Wl,--allow-shlib-undefined -o cl_sparse_sgemm -DARM_COMPUTE_CL
g++ examples/cl_sparse_sgemm.cpp utils/Utils.cpp -I. -Iinclude -std=c++11 -mfpu=neon -Lbuild -larm_compute -larm_compute_core -lpthread -Wl,--allow-shlib-undefined -o cl_sparse_sgemm -DARM_COMPUTE_CL
