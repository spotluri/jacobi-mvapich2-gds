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

#        export OMPI_MCA_btl_openib_want_cuda_gdr=0
#        export OMPI_MCA_btl_openib_cuda_async_recv=false

#        export OMPI_MCA_btl_openib_want_cuda_gdr=1

#        export OMPI_MCA_btl_openib_want_cuda_gdr=1
#        export OMPI_MCA_btl_openib_cuda_rdma_limit=$((256*1024))

#        export COMM_USE_COMM=1
#        export COMM_USE_ASYNC=1

#        export COMM_USE_COMM=1

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
