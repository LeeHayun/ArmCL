#!/bin/bash

g++ ciss.cpp ciss_support.cpp utils/Utils.cpp utils/GraphUtils.cpp utils/CommonGraphOptions.cpp -I. -Iinclude -std=c++11 -mfpu=neon -larm_compute_graph -larm_compute -larm_compute_core -Wl,--allow-shlib-undefined `pkg-config --libs opencv` -o ciss
