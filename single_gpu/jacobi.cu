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
#include <iostream>
#include <cstdio>
#include <cmath>
#include <sstream>
#include <algorithm>

#include <omp.h>

#ifdef USE_NVTX
#include <nvToolsExt.h>

const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif

#define CUDA_RT_CALL( call )                                                                        \
{                                                                                                   \
    cudaError_t cudaStatus = call;                                                                  \
    if ( cudaSuccess != cudaStatus )                                                                \
        fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n",  \
                        #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);     \
}

typedef float real;
constexpr real tol = 1.0e-8;

const real PI  = 2.0 * std::asin(1.0);

__global__ void initialize_boundaries(
    real* __restrict__ const a_new,
    real* __restrict__ const a,
    const real pi,
    const int nx, const int ny )
{
    for (int iy = blockIdx.x * blockDim.x + threadIdx.x; 
         iy < ny; 
         iy += blockDim.x * gridDim.x) {
        const real y0 = sin( 2.0 * pi * iy / (ny-1) );
        a[     iy*nx + 0 ]         = y0;
        a[     iy*nx + (nx-1) ] = y0;
        a_new[ iy*nx + 0 ]         = y0;
        a_new[ iy*nx + (nx-1) ] = y0;
    }
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

template<typename T>
T get_argval(char ** begin, char ** end, const std::string& arg, const T default_val) {
    T argval = default_val;
    char ** itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

int main(int argc, char * argv[])
{
    const int iter_max = get_argval<int>(argv, argv+argc,"-niter", 1000);
    const int nx = get_argval<int>(argv, argv+argc,"-nx", 1024);
    const int ny = get_argval<int>(argv, argv+argc,"-ny", 1024);
    
    real* a;
    real* a_new;
    
    cudaStream_t compute_stream;
    cudaStream_t push_top_stream;
    cudaStream_t push_bottom_stream;
    cudaEvent_t compute_done;
    cudaEvent_t push_top_done;
    cudaEvent_t push_bottom_done;
    
    real* l2_norm_d;
    real* l2_norm_h;
    
    int iy_start = 1;
    int iy_end = (ny-1);
    
    CUDA_RT_CALL( cudaSetDevice( 0 ) ); 
    CUDA_RT_CALL( cudaFree( 0 ) );
    
    CUDA_RT_CALL( cudaMalloc( &a, nx*ny*sizeof(real) ) );
    CUDA_RT_CALL( cudaMalloc( &a_new, nx*ny*sizeof(real) ) );
    
    CUDA_RT_CALL( cudaMemset( a, 0, nx*ny*sizeof(real) ) );
    CUDA_RT_CALL( cudaMemset( a_new, 0, nx*ny*sizeof(real) ) );
    
    //Set diriclet boundary conditions on left and right boarder
    initialize_boundaries<<<ny/128+1,128>>>( a, a_new, PI, nx, ny );
    CUDA_RT_CALL( cudaGetLastError() );
    CUDA_RT_CALL( cudaDeviceSynchronize() );
    
    CUDA_RT_CALL( cudaStreamCreate(&compute_stream) );
    CUDA_RT_CALL( cudaStreamCreate(&push_top_stream) );
    CUDA_RT_CALL( cudaStreamCreate(&push_bottom_stream) );
    CUDA_RT_CALL( cudaEventCreateWithFlags ( &compute_done, cudaEventDisableTiming ) );
    CUDA_RT_CALL( cudaEventCreateWithFlags ( &push_top_done, cudaEventDisableTiming ) );
    CUDA_RT_CALL( cudaEventCreateWithFlags ( &push_bottom_done, cudaEventDisableTiming ) );
    
    CUDA_RT_CALL( cudaMalloc( &l2_norm_d, sizeof(real) ) );
    CUDA_RT_CALL( cudaMallocHost( &l2_norm_h, sizeof(real) ) );
    
    CUDA_RT_CALL( cudaDeviceSynchronize() );
    
    printf("Jacobi relaxation: %d iterations on %d x %d mesh\n", iter_max, ny, nx);

    dim3 dim_block(32,4,1);
    dim3 dim_grid( nx/dim_block.x+1, ny/dim_block.y+1, 1 );

    int iter = 0;
    real l2_norm = 1.0;
    
    double start = omp_get_wtime();
    PUSH_RANGE("Jacobi solve",0)
    while ( l2_norm > tol && iter < iter_max )
    {
        l2_norm = 0.0;
        CUDA_RT_CALL( cudaMemsetAsync(l2_norm_d, 0 , sizeof(real), compute_stream ) );
        
        CUDA_RT_CALL( cudaStreamWaitEvent( compute_stream, push_top_done, 0 ) );
        CUDA_RT_CALL( cudaStreamWaitEvent( compute_stream, push_bottom_done, 0 ) );
        
        jacobi_kernel<<<dim_grid,dim_block,0,compute_stream>>>( a_new, a, l2_norm_d, iy_start, iy_end, nx );
        CUDA_RT_CALL( cudaGetLastError() );
        CUDA_RT_CALL( cudaEventRecord( compute_done, compute_stream ) );
        
        CUDA_RT_CALL( cudaMemcpyAsync( l2_norm_h, l2_norm_d, sizeof(real), cudaMemcpyDeviceToHost, compute_stream ) );
        
        //Apply periodic boundary conditions
        
        CUDA_RT_CALL( cudaStreamWaitEvent( push_top_stream, compute_done, 0 ) );
        CUDA_RT_CALL( cudaMemcpyAsync(
            a_new,
            a_new+(iy_end-1)*nx,
            nx*sizeof(real), cudaMemcpyDeviceToDevice, push_top_stream ) );
        CUDA_RT_CALL( cudaEventRecord( push_top_done, push_top_stream ) );
        
        CUDA_RT_CALL( cudaStreamWaitEvent( push_bottom_stream, compute_done, 0 ) );
        CUDA_RT_CALL( cudaMemcpyAsync(
            a_new+iy_end*nx,
            a_new+iy_start*nx,
            nx*sizeof(real), cudaMemcpyDeviceToDevice, compute_stream ) );
        CUDA_RT_CALL( cudaEventRecord( push_bottom_done, push_bottom_stream ) );
        
        CUDA_RT_CALL( cudaStreamSynchronize( compute_stream ) );
        l2_norm = *l2_norm_h;
        l2_norm = std::sqrt( l2_norm );
        if((iter % 100) == 0) printf("%5d, %0.6f\n", iter, l2_norm);
        
        std::swap(a_new,a);
        iter++;
    }
    POP_RANGE
    double stop = omp_get_wtime();

    printf( "%dx%d: 1 GPU: %8.4f s\n", ny,nx, (stop-start) );
    
    CUDA_RT_CALL( cudaEventDestroy( push_bottom_done ) );
    CUDA_RT_CALL( cudaEventDestroy( push_top_done ) );
    CUDA_RT_CALL( cudaEventDestroy( compute_done ) );
    CUDA_RT_CALL( cudaStreamDestroy( push_bottom_stream ) );
    CUDA_RT_CALL( cudaStreamDestroy( push_top_stream ) );
    CUDA_RT_CALL( cudaStreamDestroy( compute_stream ) );
    
    CUDA_RT_CALL( cudaFreeHost( l2_norm_h ) );
    CUDA_RT_CALL( cudaFree( l2_norm_d ) );
    
    CUDA_RT_CALL( cudaFree( a_new ) );
    CUDA_RT_CALL( cudaFree( a ) );
    CUDA_RT_CALL( cudaDeviceReset() );
    return 0;
}
