# CESpGEMM

## Requirements
```
gcc (with support for c++17)
OpenMP
MKL >= 2024.1
nvcc
cmake >= 3.28
```



## Configure

### Select CUDA Compute Capability
1. Search for the compute capability on https://developer.nvidia.com/cuda-gpus
2. Open "CMakePresets.json". Replace "CUDA_ARCH": "**75**" with "CUDA_ARCH": "**YOUR_COMPUTE_CAPABILITY**"

### Set MKL
1. Install MKL from https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
2. Open "CMakePresets.json". Fill in the "MKLROOT" field with the path to installed MKL directory.



### For Debug version: 
```shell
cmake --preset=Debug1
```
### For Release version:
```shell
cmake --preset=P1
```
### Simply Test Ptr Compression:
```shell
cmake --preset=TC1
```
## Build
### For Debug version: 
```shell
cmake --build debug1/
```
### For Release version:
```shell
cmake --build build1/
```
### Simply Test Ptr Compression:
```shell
cmake --build preproc1/
```

## Run SpGEMM

```shell
cd build1
./compute [params]
```
parameters are:
```
Required command line arguments:
  -A, Path To Matrix A (.mtx). E.g. -A path/to/a.mtx
Additional command line arguments:
  -B, Path To Matrix B (.mtx). E.g. -B path/to/b.mtx
  -ATA, Calculate A^T*A. E.g. -ATA 1
  -BA, rows in a block of A(default 0). E.g. -BA 1024
  -BB, columns in a block of B(default 0). E.g. -BB 1024
  -NW, number of cpu wokers in mkl(default 8). E.g. -NW 8
  -POOL, (Initial) PoolSize, to contain blocksize results(default 10^8), E.g -POOL 10000000
  -O, the name of the file of the SpGEMM result(in binary, default empty, i.e. not writing file), E.g -O resultA
```

If either **BA** or **BB** is set to **0**, CESpGEMM will reset (**BA**,**BB**) to (12288, 24576)


