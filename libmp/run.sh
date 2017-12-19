#!/bin/bash
#SBATCH --partition=brdw 
#SBATCH --nodes=2
#SBATCH --mincpus=1
#SBATCH --ntasks-per-node=1

. ~/work/cuda90_env.sh
#. ~/work/openmpi_2.0.2a1_env.sh
. ~/work/openmpi_3.0.0_cuda_slurm_knem_env.sh
#. ~/work/mvapich2_gdr_2.3a_env.sh
#. ~/work/mvapich2_gdr_2.3a_slurm_env.sh

set -x

hname=$(hostname)
echo $HOSTNAME

#ldd ${PWD}/jacobi


export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
export COMM_ENABLE_DEBUG=0
export MP_ENABLE_DEBUG=0
export GDS_ENABLE_DEBUG=0

#echo "PREFIX = $PREFIX"
#echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"


for EXP in $(seq 6 13); do

SIZE=$((2**$EXP))

EXE="${PWD}/../../scripts/wrapper.sh \
    ${PWD}/jacobi -niter 1000 -nccheck 100 -nx $SIZE -ny $SIZE"

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
            --mincpus=1 \
            --ntasks-per-node=1 \
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
#
#64x64: 1 GPU:   0.0363 s, 2 GPUs:   0.0558 s, speedup:     0.65, efficiency:    32.58 
#128x128: 1 GPU:   0.0442 s, 2 GPUs:   0.0573 s, speedup:     0.77, efficiency:    38.55 
#256x256: 1 GPU:   0.0589 s, 2 GPUs:   0.0642 s, speedup:     0.92, efficiency:    45.82 
#512x512: 1 GPU:   0.0917 s, 2 GPUs:   0.0816 s, speedup:     1.12, efficiency:    56.22 
#1024x1024: 1 GPU:   0.1865 s, 2 GPUs:   0.1309 s, speedup:     1.42, efficiency:    71.21 
#2048x2048: 1 GPU:   0.4866 s, 2 GPUs:   0.2865 s, speedup:     1.70, efficiency:    84.92 

#        export OMPI_MCA_btl_openib_want_cuda_gdr=1
#

#        export OMPI_MCA_btl_openib_want_cuda_gdr=1
#        export OMPI_MCA_btl_openib_cuda_rdma_limit=$((256*1024))
#
#64x64: 1 GPU:   0.0363 s, 2 GPUs:   0.0307 s, speedup:     1.18, efficiency:    58.99 
#128x128: 1 GPU:   0.0440 s, 2 GPUs:   0.0332 s, speedup:     1.33, efficiency:    66.28 
#256x256: 1 GPU:   0.0590 s, 2 GPUs:   0.0395 s, speedup:     1.49, efficiency:    74.64 
#512x512: 1 GPU:   0.0913 s, 2 GPUs:   0.0561 s, speedup:     1.63, efficiency:    81.31 
#1024x1024: 1 GPU:   0.1864 s, 2 GPUs:   0.1035 s, speedup:     1.80, efficiency:    90.03 
#2048x2048: 1 GPU:   0.4874 s, 2 GPUs:   0.2566 s, speedup:     1.90, efficiency:    94.98 
#4096x4096: 1 GPU:   1.5494 s, 2 GPUs:   0.7905 s, speedup:     1.96, efficiency:    98.00 
#8192x8192: 1 GPU:   5.5187 s, 2 GPUs:   2.7813 s, speedup:     1.98, efficiency:    99.21 
#
#64x64: 1 GPU:   0.0360 s, 2 GPUs:   0.0302 s, speedup:     1.19, efficiency:    59.54
#128x128: 1 GPU:   0.0440 s, 2 GPUs:   0.0324 s, speedup:     1.36, efficiency:    67.94
#256x256: 1 GPU:   0.0584 s, 2 GPUs:   0.0394 s, speedup:     1.48, efficiency:    74.16
#512x512: 1 GPU:   0.0918 s, 2 GPUs:   0.0556 s, speedup:     1.65, efficiency:    82.62
#1024x1024: 1 GPU:   0.2424 s, 2 GPUs:   0.1035 s, speedup:     2.34, efficiency:   117.11
#2048x2048: 1 GPU:   0.4869 s, 2 GPUs:   0.2571 s, speedup:     1.89, efficiency:    94.70
#4096x4096: 1 GPU:   1.5490 s, 2 GPUs:   0.7903 s, speedup:     1.96, efficiency:    98.00
#8192x8192: 1 GPU:   5.5190 s, 2 GPUs:   2.7823 s, speedup:     1.98, efficiency:    99.18
#
#64x64: 1 GPU:   0.0356 s, 2 GPUs:   0.0309 s, speedup:     1.15, efficiency:    57.64
#128x128: 1 GPU:   0.0439 s, 2 GPUs:   0.0323 s, speedup:     1.36, efficiency:    68.00
#256x256: 1 GPU:   0.0587 s, 2 GPUs:   0.0389 s, speedup:     1.51, efficiency:    75.38
#512x512: 1 GPU:   0.0914 s, 2 GPUs:   0.0559 s, speedup:     1.64, efficiency:    81.76
#1024x1024: 1 GPU:   0.1865 s, 2 GPUs:   0.1037 s, speedup:     1.80, efficiency:    89.98
#2048x2048: 1 GPU:   0.4869 s, 2 GPUs:   0.2568 s, speedup:     1.90, efficiency:    94.81
#4096x4096: 1 GPU:   1.5496 s, 2 GPUs:   0.7900 s, speedup:     1.96, efficiency:    98.07
#8192x8192: 1 GPU:   5.5191 s, 2 GPUs:   2.7808 s, speedup:     1.98, efficiency:    99.24
#

        export COMM_USE_COMM=1
        export COMM_USE_ASYNC=1
#64x64: 1 GPU:   0.0362 s, 2 GPUs:   0.0458 s, speedup:     0.79, efficiency:    39.47 
#128x128: 1 GPU:   0.0440 s, 2 GPUs:   0.0481 s, speedup:     0.92, efficiency:    45.76 
#256x256: 1 GPU:   0.0586 s, 2 GPUs:   0.0554 s, speedup:     1.06, efficiency:    52.86 
#512x512: 1 GPU:   0.0915 s, 2 GPUs:   0.0718 s, speedup:     1.27, efficiency:    63.68 
#1024x1024: 1 GPU:   0.1870 s, 2 GPUs:   0.1195 s, speedup:     1.57, efficiency:    78.27 
#2048x2048: 1 GPU:   0.4864 s, 2 GPUs:   0.2715 s, speedup:     1.79, efficiency:    89.59 
#4096x4096: 1 GPU:   1.5492 s, 2 GPUs:   0.8037 s, speedup:     1.93, efficiency:    96.38 
#8192x8192: 1 GPU:   5.5186 s, 2 GPUs:   2.7931 s, speedup:     1.98, efficiency:    98.79 
#64x64: 1 GPU:   0.0362 s, 2 GPUs:   0.0455 s, speedup:     0.80, efficiency:    39.76
#128x128: 1 GPU:   0.0440 s, 2 GPUs:   0.0478 s, speedup:     0.92, efficiency:    45.98
#256x256: 1 GPU:   0.0584 s, 2 GPUs:   0.0551 s, speedup:     1.06, efficiency:    53.02
#512x512: 1 GPU:   0.0911 s, 2 GPUs:   0.0718 s, speedup:     1.27, efficiency:    63.47
#1024x1024: 1 GPU:   0.1863 s, 2 GPUs:   0.1194 s, speedup:     1.56, efficiency:    78.03
#2048x2048: 1 GPU:   0.4875 s, 2 GPUs:   0.2716 s, speedup:     1.79, efficiency:    89.73
#4096x4096: 1 GPU:   1.5491 s, 2 GPUs:   0.8037 s, speedup:     1.93, efficiency:    96.38
#8192x8192: 1 GPU:   5.5190 s, 2 GPUs:   2.7929 s, speedup:     1.98, efficiency:    98.80


#        export COMM_USE_COMM=1
#
#64x64: 1 GPU:   0.0354 s, 2 GPUs:   0.0280 s, speedup:     1.26, efficiency:    63.21 
#128x128: 1 GPU:   0.0442 s, 2 GPUs:   0.0309 s, speedup:     1.43, efficiency:    71.39 
#256x256: 1 GPU:   0.0586 s, 2 GPUs:   0.0377 s, speedup:     1.56, efficiency:    77.82 
#512x512: 1 GPU:   0.0913 s, 2 GPUs:   0.0537 s, speedup:     1.70, efficiency:    85.04 
#1024x1024: 1 GPU:   0.1862 s, 2 GPUs:   0.1010 s, speedup:     1.84, efficiency:    92.17 
#2048x2048: 1 GPU:   0.4870 s, 2 GPUs:   0.2544 s, speedup:     1.91, efficiency:    95.71 
#4096x4096: 1 GPU:   1.5491 s, 2 GPUs:   0.7862 s, speedup:     1.97, efficiency:    98.52 
#8192x8192: 1 GPU:   5.5197 s, 2 GPUs:   2.7779 s, speedup:     1.99, efficiency:    99.35 
#64x64: 1 GPU:   0.0363 s, 2 GPUs:   0.0276 s, speedup:     1.31, efficiency:    65.64 
#128x128: 1 GPU:   0.0441 s, 2 GPUs:   0.0305 s, speedup:     1.45, efficiency:    72.37 
#256x256: 1 GPU:   0.0589 s, 2 GPUs:   0.0380 s, speedup:     1.55, efficiency:    77.61 
#512x512: 1 GPU:   0.0915 s, 2 GPUs:   0.0539 s, speedup:     1.70, efficiency:    84.96 
#1024x1024: 1 GPU:   0.1865 s, 2 GPUs:   0.1012 s, speedup:     1.84, efficiency:    92.11 
#2048x2048: 1 GPU:   0.4871 s, 2 GPUs:   0.2547 s, speedup:     1.91, efficiency:    95.62 
#4096x4096: 1 GPU:   1.5501 s, 2 GPUs:   0.7865 s, speedup:     1.97, efficiency:    98.55 
#8192x8192: 1 GPU:   5.5192 s, 2 GPUs:   2.7775 s, speedup:     1.99, efficiency:    99.36 
#
        if true; then
            which mpirun
            mpirun $EXE
        if false; then

            salloc \
                --nodes=2 \
                --mincpus=1 \
                --ntasks-per-node=1 \
                --partition=brdw \
                $MPI_HOME/bin/mpirun \
                $EXE
        fi

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
done
