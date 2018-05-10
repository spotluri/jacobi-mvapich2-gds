/* Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <cstdio>

#define CUDA_RT_CALL( call )                                                                        \
{                                                                                                   \
    cudaError_t cudaStatus = call;                                                                  \
    if ( cudaSuccess != cudaStatus )                                                                \
        fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n",  \
                        #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);     \
}

#ifdef USE_DOUBLE
    typedef double real;
    #define MPI_REAL_TYPE MPI_DOUBLE
#else
    typedef float real;
    #define MPI_REAL_TYPE MPI_FLOAT
#endif

__global__ void initialize_boundaries(
    real* __restrict__ const a_new,
    real* __restrict__ const a,
    const real pi,
    const int offset,
    const int nx, const int my_ny, const int ny )
{
    for (int iy = blockIdx.x * blockDim.x + threadIdx.x; 
         iy < my_ny; 
         iy += blockDim.x * gridDim.x) {
        const real y0 = sin( 2.0 * pi * (offset+iy) / (ny-1) );
        a[     iy*nx + 0 ]         = y0;
        a[     iy*nx + (nx-1) ] = y0;
        a_new[ iy*nx + 0 ]         = y0;
        a_new[ iy*nx + (nx-1) ] = y0;
    }
}

void launch_initialize_boundaries(
    real* __restrict__ const a_new,
    real* __restrict__ const a,
    const real pi,
    const int offset,
    const int nx, const int my_ny, const int ny )
{
    initialize_boundaries<<<my_ny/128+1,128>>>( a_new, a, pi, offset, nx, my_ny, ny );
    CUDA_RT_CALL( cudaGetLastError() );
}

__global__ void jacobi_kernel(
          real* __restrict__ const a_new,
    const real* __restrict__ const a,
          real* __restrict__ const l2_norm,
    const int iy_start, const int iy_end,
    const int nx)
{
    for (int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start; 
         iy < iy_end; 
         iy += blockDim.y * gridDim.y) {
    for (int ix = blockIdx.x * blockDim.x + threadIdx.x + 1; 
         ix < (nx-1); 
         ix += blockDim.x * gridDim.x) {
        const real new_val = 0.25 * ( a[ iy * nx + ix + 1 ] + a[ iy * nx + ix - 1 ]
                                    + a[ (iy+1) * nx + ix ] + a[ (iy-1) * nx + ix ] );
        a_new[ iy * nx + ix ] = new_val;
        real residue = new_val - a[ iy * nx + ix ];
        atomicAdd( l2_norm, residue*residue );
    }}
}

void launch_jacobi_kernel(
          real* __restrict__ const a_new,
    const real* __restrict__ const a,
          real* __restrict__ const l2_norm,
    const int iy_start, const int iy_end,
    const int nx,
    cudaStream_t stream)
{
    dim3 dim_block(32,4,1);
    dim3 dim_grid( nx/dim_block.x+1, (iy_end-iy_start)/dim_block.y+1, 1 );
    jacobi_kernel<<<dim_grid,dim_block,0,stream>>>( a_new, a, l2_norm, iy_start, iy_end, nx );
    CUDA_RT_CALL( cudaGetLastError() );
}

__global__ void dummy_kernel(uint *ptr)
{
    if (threadIdx.x > 1025) {
        *ptr = threadIdx.x;
    }
}

void launch_dummy_kernel(cudaStream_t stream)
{
    dim3 dim_block(32,1,1);
    dim3 dim_grid(1,1,1);
    dummy_kernel<<<dim_grid,dim_block,0,stream>>>(NULL);
    CUDA_RT_CALL( cudaGetLastError() );
}

/*
 ******************************************************
 * Async KI Model - communication kernel
 ******************************************************
*/

#include <comm.h>

// ==================== Init ====================
#define TOT_SCHEDS 128
const int max_scheds = TOT_SCHEDS;
const int max_types = 3;
static int n_scheds = TOT_SCHEDS;

typedef struct sched_info {
//  mp::mlx5::gdsync::sem32_t sema;
	unsigned int block;
	unsigned int done[max_types];
} sched_info_t;

__device__ sched_info_t scheds[max_scheds];

__global__ void scheds_init()
{
	int j = threadIdx.x;
	assert(gridDim.x == 1);
	assert(blockDim.x >= max_scheds);
	if (j < max_scheds) {
 //   scheds[j].sema.sem = 0;
//    scheds[j].sema.value = 1;
		scheds[j].block = 0;
		for (int i = 0; i < max_types; ++i)
			scheds[j].done[i] = 0;
	}
}

__device__ static inline unsigned int elect_block(sched_info &sched)
{
	unsigned int ret;
	const int n_grids = gridDim.x * gridDim.y * gridDim.z;
	__shared__ unsigned int block;
	if (0 == threadIdx.x && 0 == threadIdx.y && 0 == threadIdx.z) {
		// 1st guy gets 0
		block = atomicInc(&sched.block, n_grids);
	}
	__syncthreads();
	ret = block;
	return ret;
}

// ==================== Comm Kernel ====================
__global__ void comm_kernel(int sched_id, comm_dev_descs_t descs)
{
	assert(sched_id >= 0 && sched_id < max_scheds);
	sched_info_t &sched = scheds[sched_id];
	int block = elect_block(sched);

	//First one receive
	if(block == 0)
	{
		assert(blockDim.x >= descs->n_wait);
		if (threadIdx.x < descs->n_wait) {
			//printf("blockDim.x: %d blockIdx.x: %d, threadIdx.x %d waiting\n", blockDim.x, blockIdx.x, threadIdx.x);
			mp::device::mlx5::wait(descs->wait[threadIdx.x]);
			// write MP trk flag
			// note: no mem barrier here!!!
			mp::device::mlx5::signal(descs->wait[threadIdx.x]);
		}

		__syncthreads();
	}

	//Second one send
	if(block == 1)
	{
		//n_ready
		if (threadIdx.x < descs->n_tx) {
			// wait for ready
			gdsync::device::wait_geq(descs->ready[threadIdx.x]);
			// signal NIC
			mp::device::mlx5::send(descs->tx[threadIdx.x]);				
		}
		__syncthreads();
	}
}


void launch_comm_kernel(comm_dev_descs_t descs, cudaStream_t stream, int num_ranks)
{
	dim3 dim_block( num_ranks,1,1);
	dim3 dim_grid(2,1,1);
	//assert( descs->n_ready > 0 );

	if (n_scheds >= max_scheds) {
		scheds_init<<<1, max_scheds, 0, stream>>>();
		n_scheds = 0;
	}

	comm_kernel<<<dim_grid,dim_block,0,stream>>>(n_scheds++, descs);
	CUDA_RT_CALL( cudaGetLastError() );
}

__global__ void jacobi_kernel_comm(
	real* __restrict__ const a_new,
	const real* __restrict__ const a,
	real* __restrict__ const l2_norm,
	const int iy_start, const int iy_end,
	const int nx,
	comm_dev_descs_t descs,
	int sched_id)
{
	for (int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start; 
		iy < iy_end; 
		iy += blockDim.y * gridDim.y) {
		for (int ix = blockIdx.x * blockDim.x + threadIdx.x + 1; 
			ix < (nx-1); 
			ix += blockDim.x * gridDim.x) {
			const real new_val = 0.25 * ( a[ iy * nx + ix + 1 ] + a[ iy * nx + ix - 1 ]
				+ a[ (iy+1) * nx + ix ] + a[ (iy-1) * nx + ix ] );
			a_new[ iy * nx + ix ] = new_val;
			real residue = new_val - a[ iy * nx + ix ];
			atomicAdd( l2_norm, residue*residue );
	}}

	assert(sched_id >= 0 && sched_id < max_scheds);
	sched_info_t &sched = scheds[sched_id];
	int gridSize = gridDim.x * gridDim.y * gridDim.z;
	int block = elect_block(sched);
	
	//First one receive
	if(block == 0)
	{
		if (threadIdx.x == 0 &&  threadIdx.y < descs->n_wait)
		{
			mp::device::mlx5::wait(descs->wait[threadIdx.y]);
			// write MP trk flag
			// note: no mem barrier here!!!
			mp::device::mlx5::signal(descs->wait[threadIdx.y]);
		}
		__syncthreads();
	}

	//Last one sends
	if(block == (gridSize-1))
	{
		if (threadIdx.x == 0 &&  threadIdx.y < descs->n_tx)
		{
			// wait for ready
			gdsync::device::wait_geq(descs->ready[threadIdx.y]);
			// signal NIC
			mp::device::mlx5::send(descs->tx[threadIdx.y]);				
		}

		__syncthreads();
	}
}

void launch_jacobi_comm_kernel(
	real* __restrict__ const a_new,
	const real* __restrict__ const a,
	real* __restrict__ const l2_norm,
	const int iy_start, const int iy_end,
	const int nx,
	cudaStream_t stream,
	comm_dev_descs_t descs)
{
	if (n_scheds >= max_scheds) {
		scheds_init<<<1, max_scheds, 0, stream>>>();
		n_scheds = 0;
	}

	dim3 dim_block(32,4,1);
	dim3 dim_grid( nx/dim_block.x+1, (iy_end-iy_start)/dim_block.y+1, 1 );
	
	jacobi_kernel_comm<<<dim_grid,dim_block,0,stream>>>( a_new, a, l2_norm, 
							iy_start, iy_end, nx,
							descs, n_scheds++);
	CUDA_RT_CALL( cudaGetLastError() );
}
