cmake --preset=P1
cmake --preset=P2
cmake --preset=P3
cmake --preset=P4
cmake --preset=Debug1
cmake --preset=Debug2
cmake --preset=Debug3
cmake --preset=Debug4

cmake --build ./build_1GPU
cmake --build ./build_2GPU
cmake --build ./build_3GPU
cmake --build ./build_4GPU
cmake --build ./debug_1GPU
cmake --build ./debug_2GPU
cmake --build ./debug_3GPU
cmake --build ./debug_4GPU

