cmake_minimum_required(VERSION 3.29)
project(CESpGEMM LANGUAGES CUDA CXX)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


if (CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -O0")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG -O0")
endif()

if (DEFINED CUDA_ARCH)
else()
  message(FATAL_ERROR "did not define CUDA_ARCH")
endif()

include_directories(includes)
include_directories(/usr/local/cuda/include/)
include_directories(/usr/local/cuda-12.1/include)
link_directories(/lib64)

set(SpGEMM_HEADERS
  # includes/TaskManage.h
  includes/CSR.h
  includes/FileIO.h
  includes/helper.cuh
  # includes/StaticVector.cuh
  includes/Storage.h
  includes/CSR.h
  # includes/computeGPU.h
  # includes/OmpExec.h
  # includes/Runner.cuh
  includes/pipeline.h
  # includes/pipeline-worker-impl.h
  includes/pipelined-scheme.h
  includes/WritableStorage.h
  includes/task_alloc.h
  # includes/cpu_task_fetcher.h
  includes/cpu_task_fetcher_row.h
  includes/gpu_task_fetcher_async.h
  includes/prepare.h
  includes/compute-dispatch.h
  includes/merge.h
  includes/ThreadSafeQueue.h
  includes/computeGPUKernel.cuh
  includes/profiler.h
)

set(SpGEMM_SOURCES
  # sources/ExpandMergeKnl.cu
  sources/Storage.cpp
  # sources/OmpExec.cpp
  sources/CSR.cpp
  sources/FileIO.cpp
  # sources/computeGPU.cu
  # sources/computeCPU.cpp
  # sources/computeGPUKernel.cu
  sources/compressor.cpp
  sources/decomp.cu
  sources/computeGPUAsync.cu
  sources/compress4B.cpp
  sources/profiler.cu
  sources/FileWrite.cpp
)

set(MKL_SOURCE
  sources/computeMKL.cpp
)
# set(MKLROOT
#   /home/zhangcheng/intel/oneapi/mkl/2024.1
# )

set(MAIN_SOURCES
  sources/main.cpp
)

set(TEST_COMP_SOURCE
  sources/testComp.cu
)



add_executable(compute ${MAIN_SOURCES})
# add_executable(io ${MAIN_SOURCES})
add_library(CESpGEMM ${SpGEMM_HEADERS} ${SpGEMM_SOURCES})
add_library(MKL_compute ${SpGEMM_HEADERS} ${MKL_SOURCE})

add_executable(testComp ${TEST_COMP_SOURCE})

# add_library(IO_runner_min ${PIPELINE_WORKER_HEADERS})
# add_library(IO_runner_full ${PIPELINE_WORKER_HEADERS})

message("cuda flags are ${CMAKE_CUDA_FLAGS}")

set_target_properties(compute PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CUDA_ARCHITECTURES ${CUDA_ARCH}
)

# set_target_properties(io PROPERTIES
#   CXX_STANDARD 17
#   CXX_STANDARD_REQUIRED ON
#   CUDA_ARCHITECTURES ${CUDA_ARCH}
# )

set_target_properties(MKL_compute PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  LINKER_LANGUAGE CXX
)

set_target_properties(CESpGEMM PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CUDA_ARCHITECTURES ${CUDA_ARCH}
  LINKER_LANGUAGE CXX
)


set_target_properties(testComp PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CUDA_ARCHITECTURES ${CUDA_ARCH}
  LINKER_LANGUAGE CXX
)

target_link_options(MKL_compute PUBLIC
  -m64
  -L${MKLROOT}/lib/intel64
  -Wl,--no-as-needed
  -lmkl_intel_lp64
  -lmkl_gnu_thread
  -lmkl_core
  -lgomp
  -lpthread
  -lm
  -ldl
  -I${MKLROOT}/include
)
target_compile_options(testComp PUBLIC
	-Xcompiler -fopenmp
)
# set_target_properties(IO_runner_min PROPERTIES
#   CXX_STANDARD 17
#   CXX_STANDARD_REQUIRED ON
#   CUDA_ARCHITECTURES ${CUDA_ARCH}
#   LINKER_LANGUAGE CXX
# )
  
# set_target_properties(IO_runner_full PROPERTIES
#   CXX_STANDARD 17
#   CXX_STANDARD_REQUIRED ON
#   CUDA_ARCHITECTURES ${CUDA_ARCH}
#   LINKER_LANGUAGE CXX
# )




find_package(OpenMP REQUIRED)
# target_link_libraries(IO_runner_min PRIVATE OpenMP::OpenMP_CXX)
# target_link_libraries(IO_runner_full PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(CESpGEMM PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(compute PUBLIC OpenMP::OpenMP_CXX CESpGEMM MKL_compute)

target_link_libraries(testComp PUBLIC OpenMP::OpenMP_CXX CESpGEMM)
# target_link_libraries(io PRIVATE OpenMP::OpenMP_CXX CESpGEMM)



if (NUM_DEVICES)
  target_compile_definitions(CESpGEMM PUBLIC NUM_DEVICES=${NUM_DEVICES})
  # target_compile_definitions(IO_runner_min PUBLIC NUM_DEVICES=${NUM_DEVICES})
  # target_compile_definitions(IO_runner_full PUBLIC NUM_DEVICES=${NUM_DEVICES} FILEWRITE=1)
else ()
  message(FATAL_ERROR "did not define 'NUM_DEVICES'")
endif()

if(TEST_PREPROC)
target_compile_definitions(CESpGEMM PUBLIC TEST_PREPROC=1)
  target_compile_definitions(compute PUBLIC TEST_PREPROC=1)
endif()
