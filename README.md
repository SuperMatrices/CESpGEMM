# CESpGEMM

CESpGEMM is a high-performance Sparse General Matrix-Matrix Multiplication (SpGEMM) library that leverages CPU-GPU heterogeneous computing with automatic parameter tuning capabilities.

## Features

- **Multi-GPU Support**: Scalable execution across 1-4 NVIDIA GPUs
- **Automatic Tuning**: Genetic algorithm and random search for optimal parameters
- **Pointer Compression**: Memory-efficient storage for sparse matrices
- **Hybrid Computing**: CPU-GPU pipeline with OpenMP parallelization
- **Matrix Market Format**: Support for `.mtx` file input/output
- **Flexible Configuration**: Multiple build presets for different use cases

## Requirements

```
gcc (with support for C++17)
OpenMP
CUDA nvcc
cmake >= 3.29
Intel MKL(2024.1)
```

## Quick Start

### 1. Configure CUDA Compute Capability

Find your GPU's compute capability at https://developer.nvidia.com/cuda-gpus and update `CMakePresets.json`:

```json
"cacheVariables": {
  "GPU_ARCH": "YOUR_COMPUTE_CAPABILITY"  // e.g., 75, 80, 86, 90
}
```

### 2. Configure Build

Choose a preset based on your needs:

**Release Builds (1-4 GPUs):**
```shell
cmake --preset=P1    # 1 GPU
cmake --preset=P2    # 2 GPUs
cmake --preset=P3    # 3 GPUs
cmake --preset=P4    # 4 GPUs
```

**Debug Builds (1-4 GPUs):**
```shell
cmake --preset=Debug1    # 1 GPU
cmake --preset=Debug2    # 2 GPUs
cmake --preset=Debug3    # 3 GPUs
cmake --preset=Debug4    # 4 GPUs
```

**Test Pointer Compression:**
```shell
cmake --preset=TC1
```

### 3. Build

```shell
# For P1/P2/P3/P4:
cmake --build build_1GPU/    # or build_2GPU/, build_3GPU/, build_4GPU/

# For Debug1/Debug2/Debug3/Debug4:
cmake --build debug_1GPU/    # or debug_2GPU/, debug_3GPU/, debug_4GPU/

# For TC1:
cmake --build preproc1/
```

## Usage

### Basic SpGEMM Computation

```shell
cd build_1GPU
./compute [options]
```

### Required Arguments

| Option | Description | Example |
|--------|-------------|---------|
| `-A` | Path to Matrix A (.mtx file) | `-A path/to/a.mtx` |

### Optional Arguments

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `-B` | Path to Matrix B (.mtx file) | Same as A | `-B path/to/b.mtx` |
| `-ATA` | Calculate A^T*A | 0 | `-ATA 1` |
| `-BA` | Rows in a block of A | 4096 | `-BA 4096` |
| `-BB` | Columns in a block of B | 12288 | `-BB 8192` |
| `-NW` | Number of OpenMP workers | 8 | `-NW 16` |
| `-POOL` | Pool size for block results | 1e8 | `-POOL 10000000` |
| `-O` | Output file name (binary format) | (no output) | `-O result.bin` |
| `-DEVICES` | GPU devices to use | "0" | `-DEVICES "0,1,2"` |
| `-MAX_GMEM` | GPU memory limit in GB | -1 (-1 means no limit) | `-MAX_GMEM 24` |
| `-DEBUG` | Runtime debug level | 0 | `-DEBUG 3` |
| `-SKIP` | Skip computation (initialize only) | 0 | `-SKIP 1` |

### Tuning Parameters (Advanced)

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `-GIVE` | Use given parameters instead of auto-tuning | 0 | `-GIVE 1` |
| `-NZHEAD` | Max zero heads in compression | 4 | `-NZHEAD 8` |
| `-ZLEN` | Minimum consecutive zero length in compression | 16 | `-ZLEN 32` |
| `-SLEN` | Segment length in compression | 512 | `-SLEN 256` |
| `-GRATIO` | GPU ratio percentage | 3 | `-GRATIO 50` |
| `-RAND` | Use random search only (no genetic algorithm) | 0 | `-RAND 1` |

### Examples

**Basic SpGEMM with auto-tuning:**
```shell
./compute -A matrices/a.mtx
```

**SpGEMM with two different matrices:**
```shell
./compute -A matrices/a.mtx -B matrices/b.mtx
```

**Compute with custom block sizes:**
```shell
./compute -A matrices/a.mtx -BA 8192 -BB 16384
```

**Use specific GPUs and save results:**
```shell
./compute -A matrices/a.mtx -DEVICES "0,1" -O my_result
```

**Run with given tuning parameters:**
```shell
./compute -A matrices/a.mtx -GIVE 1 -BA 4096 -BB 8192 -NZHEAD 4 -ZLEN 16 -SLEN 512 -GRATIO 10
```

## Build Presets

### Release Presets (P1-P4)

Optimized builds for production use with different GPU configurations:
- **P1**: 1 GPU, DEBUG_LEVEL=1
- **P2**: 2 GPUs, DEBUG_LEVEL=3
- **P3**: 3 GPUs, DEBUG_LEVEL=3
- **P4**: 4 GPUs, DEBUG_LEVEL=3

### Debug Presets (Debug1-Debug4)

Debug builds with additional error checking:
- **Debug1**: 1 GPU, DEBUG_LEVEL=3
- **Debug2**: 2 GPUs, DEBUG_LEVEL=3
- **Debug3**: 3 GPUs, DEBUG_LEVEL=3
- **Debug4**: 4 GPUs, DEBUG_LEVEL=3

### Test Preset (TC1)

Special build for testing pointer compression functionality.

## Output

### Console Output

The program prints profiling statistics including:
- CPU preprocessing time and compression ratio
- GPU kernel execution time and memory transfer
- Overall pipeline throughput
- Number of blocks processed by CPU vs GPU

### File Output

When `-O` is specified, the program generates:
- `<name>.ptr`: Pointer array for the result matrix
- `<name>.val`: Value array for the result matrix (binary format)

## Project Structure

```
CESpGEMM/
├── CMakeLists.txt          # Build configuration
├── CMakePresets.json       # CMake presets for different configurations
├── includes/               # Header files
│   ├── CSR.h              # CSR matrix format
│   ├── Storage.h          # Global storage management
│   ├── pipeline.h         # CPU-GPU pipeline
│   ├── autotune.h         # Genetic algorithm tuning
│   └── ...
├── sources/                # Source files
│   ├── main.cpp           # Main entry point
│   ├── CSR.cpp            # CSR implementation
│   ├── computeGPUAsync.cu # GPU computation
│   └── ...
└── README.md
```

