#!/bin/bash

#. ~/work/cuda90_env.sh
#. ~/work/openmpi_2.0.2a1_env.sh
#. ~/work/mvapich2_gdr_2.3a_env.sh
#. ~/work/mvapich2_gdr_2.3a_slurm_env.sh

#set -x

#ldd ${PWD}/jacobi


export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
export COMM_ENABLE_DEBUG=0
export MP_ENABLE_DEBUG=0
export GDS_ENABLE_DEBUG=0

#echo "PREFIX = $PREFIX"
#echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"

EXE="${PWD}/../../scripts/wrapper.sh \
    ${PWD}/jacobi -niter 100"

echo "MPI is $MPI_NAME"

case "$MPI_NAME" in
    mvapich2)

        export MV2_USE_CUDA=1
        export MV2_USE_GPUDIRECT=0
        export MV2_USE_GPUDIRECT_GDRCOPY=0
        export MV2_USE_GPUDIRECT_LOOPBACK=0
        export MV2_GPUDIRECT_LIMIT=$((1024*128))
        export MV2_USE_GPUDIRECT_RECEIVE_LIMIT=$((1024*1024*256))
        #export MV2_GPUDIRECT_GDRCOPY_LIB=$PREFIX/lib/libgdrapi.so

        srun \
            --nodes=2 \
            --partition=brdw \
            $EXE

        ;;

    openmpi)

# debugging helpers
#export OMPI_MCA_orte_base_help_aggregate=0
#export OMPI_MCA_mpi_common_cuda_verbose=100
#export OMPI_MCA_opal_cuda_verbose=10

#
# OMPI 3.0.0
#
# default application clocks
# [1] GPU0 name=Tesla P100-PCIE-16GB clockRate=1328500 multiProcessorCount=56 <==
# [0] GPU0 name=Tesla P100-PCIE-16GB clockRate=1328500 multiProcessorCount=56 <==
#
#        export OMPI_MCA_btl_openib_want_cuda_gdr=0
#1024x1024: 1 GPU:   0.0173 s, 2 GPUs:   0.0133 s, speedup:     1.29, efficiency:    64.75
#1024x1024: 1 GPU:   0.0173 s, 2 GPUs:   0.0133 s, speedup:     1.31, efficiency:    65.32
#1024x1024: 1 GPU:   0.0173 s, 2 GPUs:   0.0133 s, speedup:     1.30, efficiency:    64.97
#
#        export OMPI_MCA_btl_openib_want_cuda_gdr=1
#1024x1024: 1 GPU:   0.0172 s, 2 GPUs:   0.0108 s, speedup:     1.59, efficiency:    79.70 
#1024x1024: 1 GPU:   0.0173 s, 2 GPUs:   0.0109 s, speedup:     1.59, efficiency:    79.36
#1024x1024: 1 GPU:   0.0173 s, 2 GPUs:   0.0108 s, speedup:     1.60, efficiency:    79.81
#
#        export OMPI_MCA_btl_openib_want_cuda_gdr=1
#        export OMPI_MCA_btl_openib_cuda_rdma_limit=$((256*1024))
# with streamSync
#1024x1024: 1 GPU:   0.0173 s, 2 GPUs:   0.0130 s, speedup:     1.33, efficiency:    66.57 
#1024x1024: 1 GPU:   0.0173 s, 2 GPUs:   0.0161 s, speedup:     1.07, efficiency:    53.50 
#1024x1024: 1 GPU:   0.0170 s, 2 GPUs:   0.0150 s, speedup:     1.14, efficiency:    56.89 
#1024x1024: 1 GPU:   0.0173 s, 2 GPUs:   0.0132 s, speedup:     1.31, efficiency:    65.53 
#1024x1024: 1 GPU:   0.0172 s, 2 GPUs:   0.0134 s, speedup:     1.29, efficiency:    64.27 
# with eventSync
#1024x1024: 1 GPU:   0.0171 s, 2 GPUs:   0.0156 s, speedup:     1.10, efficiency:    54.83 
#1024x1024: 1 GPU:   0.0171 s, 2 GPUs:   0.0132 s, speedup:     1.30, efficiency:    64.82
#1024x1024: 1 GPU:   0.0172 s, 2 GPUs:   0.0132 s, speedup:     1.31, efficiency:    65.49 
#1024x1024: 1 GPU:   0.0171 s, 2 GPUs:   0.0131 s, speedup:     1.30, efficiency:    64.98 
#1024x1024: 1 GPU:   0.0171 s, 2 GPUs:   0.0130 s, speedup:     1.32, efficiency:    65.78 
#
        export COMM_USE_GPU_COMM=1
#
#1024x1024: 1 GPU:   0.0173 s, 2 GPUs:   0.0117 s, speedup:     1.47, efficiency:    73.67 
#1024x1024: 1 GPU:   0.0172 s, 2 GPUs:   0.0118 s, speedup:     1.46, efficiency:    73.20 
#1024x1024: 1 GPU:   0.0175 s, 2 GPUs:   0.0117 s, speedup:     1.49, efficiency:    74.69
# cpupower frequency-set -g performance
#1024x1024: 1 GPU:   0.0171 s, 2 GPUs:   0.0116 s, speedup:     1.47, efficiency:    73.53
#1024x1024: 1 GPU:   0.0171 s, 2 GPUs:   0.0117 s, speedup:     1.46, efficiency:    73.21
#1024x1024: 1 GPU:   0.0171 s, 2 GPUs:   0.0117 s, speedup:     1.46, efficiency:    73.03 
# moving comm_flush() out of loop
#1024x1024: 1 GPU:   0.0171 s, 2 GPUs:   0.0120 s, speedup:     1.42, efficiency:    71.12
#1024x1024: 1 GPU:   0.0169 s, 2 GPUs:   0.0120 s, speedup:     1.41, efficiency:    70.37 
#1024x1024: 1 GPU:   0.0169 s, 2 GPUs:   0.0120 s, speedup:     1.41, efficiency:    70.57 

        if true; then

            salloc \
                --nodes=2 \
                --mincpus=1 \
                --ntasks-per-node=1 \
                --partition=brdw \
                $MPI_HOME/bin/mpirun \
                $EXE

#    $MPI_HOME/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw D D
#               --ntasks-per-node=2 \
#    --cpus-per-task=1 \
#    --mincpus=1 \
#    --time=00:10:00 \
#    
#    $MPI_HOME/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw D D
#
#    $MPI_HOME/libexec/osu-micro-benchmarks/get_local_rank \
#    ${PWD}/jacobi -niter 100 

        else

            ${MPI_HOME}/bin/mpirun \
                -x LD_LIBRARY_PATH \
                -mca pml ob1 \
                -mca btl openib,vader,self \
                --map-by node --report-bindings --bind-to core \
                --hostfile hosts \
                --np 2 \
                $EXE

        fi
        ;;

    *)
        echo "unsupported MPI"
        ;;
    
esac