# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
NP ?= 1
NVCC=nvcc
MPICXX=$(MPI_HOME)/bin/mpicxx
GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35
GENCODE_SM37	:= -gencode arch=compute_37,code=sm_37
GENCODE_SM50	:= -gencode arch=compute_50,code=sm_50
GENCODE_SM52	:= -gencode arch=compute_52,code=sm_52
GENCODE_SM60	:= -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60
GENCODE_SM61	:= -gencode arch=compute_61,code=sm_61
GENCODE_SM70	:= -gencode arch=compute_70,code=sm_70
GENCODE_FLAGS	:= $(GENCODE_SM35) $(GENCODE_SM60) $(GENCODE_SM70)
NVCC_FLAGS = -lineinfo $(GENCODE_FLAGS) -std=c++11 -I$(PREFIX)/include -I$(MPI_HOME)/include
MPICXX_FLAGS = -I. -I$(PREFIX)/include -I$(CUDA_HOME)/include -std=c++11  -g -D_ENABLE_CUDA_=1
#MPICXX_FLAGS += -DUSE_NVTX 
MPICXX_FLAGS += -DUSE_COMM 
LD_FLAGS = -L$(PREFIX)/lib -lmpcomm -lmp -lgdsync -lgdrapi -L$(CUDA_HOME)/lib64 -lcudart -lnvToolsExt -lcuda

all: Makefile jacobi #sndrcv

sndrcv: sndrcv.cpp
	$(MPICXX) $(MPICXX_FLAGS) $^ -L$(CUDA_HOME)/lib64 -lcudart -L$(MPI_HOME)/lib64 -o $@

osu_latency: osu_latency.o osu_pt2pt.o
	$(MPICXX) $(MPICXX_FLAGS) $^ -L$(CUDA_HOME)/lib64 -lcudart -L$(MPI_HOME)/lib64 -o $@

jacobi: jacobi.cpp jacobi_kernels.o
	$(MPICXX) $(MPICXX_FLAGS) $^ $(LD_FLAGS) -o $@

jacobi_kernels.o: jacobi_kernels.cu
	$(NVCC) $(NVCC_FLAGS) jacobi_kernels.cu -c -o $@

%.o: %.c
	$(MPICXX) $(MPICXX_FLAGS) $^ -c -o $@


.PHONY: clean

clean:
	rm -f jacobi *.o *.nvprof

memcheck: jacobi
	mpirun -np $(NP) cuda-memcheck ./jacobi

run: jacobi
	mpirun -np $(NP) ./jacobi

profile: jacobi
	mpirun -np $(NP) nvprof -o jacobi.%q{OMPI_COMM_WORLD_RANK}.nvprof --process-name "rank %q{OMPI_COMM_WORLD_RANK}" --context-name "rank %q{OMPI_COMM_WORLD_RANK}"  ./jacobi -niter 10
