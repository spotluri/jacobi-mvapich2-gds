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
#include "mpi.h"
#include "shmem.h"
#include "shmem_device.h"
#include <assert.h>

#define MPI_CALL( call ) \
{   \
    int mpi_status = call; \
    if ( 0 != mpi_status ) \
    { \
        char mpi_error_string[MPI_MAX_ERROR_STRING]; \
        int mpi_error_string_length = 0; \
        MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
        if ( NULL != mpi_error_string ) \
            fprintf(stderr, "ERROR: MPI call \"%s\" in line %d of file %s failed with %s (%d).\n", #call, __LINE__, __FILE__, mpi_error_string, mpi_status); \
        else \
            fprintf(stderr, "ERROR: MPI call \"%s\" in line %d of file %s failed with %d.\n", #call, __LINE__, __FILE__, mpi_status); \
    } \
}

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
    const int offset,
    const int nx, const int my_ny, int ny )
{
    for (int iy = blockIdx.x * blockDim.x + threadIdx.x; 
         iy < my_ny; 
         iy += blockDim.x * gridDim.x) {
         const real y0 = sin( 2.0 * pi * (offset+iy) / (ny-1) );
         a[     (iy+1)*nx + 0 ]         = y0;
         a[     (iy+1)*nx + (nx-1) ] = y0;
         a_new[ (iy+1)*nx + 0 ]         = y0;
         a_new[ (iy+1)*nx + (nx-1) ] = y0;
    }
}

__global__ void jacobi_kernel(
          real* __restrict__ const a_new,
    const real* __restrict__ const a,
          real* __restrict__ const l2_norm,
    const int iy_start, const int iy_end,
    const int nx,
    int top_pe,
    const int top_iy,
    int bottom_pe,
    const int bottom_iy 
    )
{
    for (int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start; 
         iy <= iy_end; 
         iy += blockDim.y * gridDim.y) {
    for (int ix = blockIdx.x * blockDim.x + threadIdx.x + 1; 
         ix < (nx-1); 
         ix += blockDim.x * gridDim.x) {

        const real new_val = 0.25 * ( a[ iy * nx + ix + 1 ] + a[ iy * nx + ix - 1 ]
                                    + a[ (iy+1) * nx + ix ] + a[ (iy-1) * nx + ix ] );
        a_new[ iy * nx + ix ] = new_val;

        if ( iy_start == iy )
        {
            shmem_float_p(a_new + top_iy*nx + ix, new_val, top_pe);
        }
        if ( iy_end == iy )
        {
            shmem_float_p(a_new + bottom_iy*nx + ix, new_val, bottom_pe);
        }

        real residue = new_val - a[ iy * nx + ix ];
        atomicAdd( l2_norm, residue*residue );
    }}
}

double single_gpu(const int nx, const int ny, const int iter_max, real* const a_ref_h, const int nccheck, const bool print, int mype);

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

bool get_arg(char ** begin, char ** end, const std::string& arg) {
    char ** itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}


int main(int argc, char * argv[])
{
    const int iter_max = get_argval<int>(argv, argv+argc,"-niter", 1000);
    const int nx = get_argval<int>(argv, argv+argc,"-nx", 1024);
    int ny = get_argval<int>(argv, argv+argc,"-ny", 1024);
    const int nccheck = get_argval<int>(argv, argv+argc,"-nccheck", 1);
    const bool csv = get_arg(argv, argv+argc,"-csv");
    
    real* a_new;
    
    real* a_ref_h;
    real* a_h;
    double runtime_serial = 0.0;
    
    real l2_norm = 1.0;

    int rank = 0, size = 1;
    MPI_CALL(MPI_Init(&argc, &argv));
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));
 
    int num_devices;
    CUDA_RT_CALL( cudaGetDeviceCount( &num_devices ) );
    if (num_devices < size) { 
        printf("the test requires device count >= process count \n");
        exit(-1);
    } 

    int local_rank = -1, local_size = 1;
    {
        MPI_Comm local_comm;
        MPI_Info info;
        MPI_CALL( MPI_Info_create(&info) );
        MPI_CALL( MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, info, &local_comm) );

        MPI_CALL( MPI_Comm_rank(local_comm,&local_rank) );
        MPI_CALL( MPI_Comm_size(local_comm,&local_size) );
        if (local_size < size) { 
            printf("this test works only within a node \n");
            exit(-1);
        }

        MPI_CALL( MPI_Comm_free(&local_comm) );
        MPI_CALL( MPI_Info_free(&info) );
    }
    CUDA_RT_CALL( cudaSetDevice( local_rank ) );
    CUDA_RT_CALL( cudaFree( 0 ) );

    start_pes();     
    int npes = shmem_n_pes();
    int mype = shmem_my_pe();

    shmem_barrier_all();

    bool result_correct = true;
    {
        real* a;
        
        cudaStream_t compute_stream;
        cudaEvent_t compute_done;
        
        real* l2_norm_d;
        real* l2_norm_h;
	
	//adjusting ny to equally divisible acrros PEs
	if (!mype && ny%npes && !csv) printf("increasing ny by %d to be equally divisible among PEs \n", (npes - ny%npes));
	ny = ny + (npes - ny%npes);

        // Ensure correctness if ny%size != 0
        int chunk_size = std::ceil( (1.0*ny)/npes );

	//data size has to be symmetric on all PEs in SHMEM version
	assert(chunk_size == (1.0*ny/npes));
      
        CUDA_RT_CALL( cudaMallocHost( &a_ref_h, nx*(ny+2)*sizeof(real) ) );
        CUDA_RT_CALL( cudaMallocHost( &a_h, nx*(ny+2)*sizeof(real) ) );
        runtime_serial = single_gpu(nx,ny,iter_max,a_ref_h,nccheck,!csv&&(0==mype),mype);

        shmem_barrier_all();
 
        a = (real *) shmalloc(nx*(chunk_size+2)*sizeof(real));
        a_new = (real *) shmalloc(nx*(chunk_size+2)*sizeof(real));

        cudaMemset( a, 0, nx*(chunk_size+2)*sizeof(real) );
        cudaMemset( a_new, 0, nx*(chunk_size+2)*sizeof(real) );

        //Calculate local domain boundaries
        int iy_start_global = mype * chunk_size + 1;
        int iy_end_global = iy_start_global + chunk_size - 1;

        int iy_start = 1;
        int iy_end = chunk_size;

        //calculate boundary indices for top and bottom boundaries
        int top = mype > 0 ? mype - 1 : (npes-1);
        int bottom = (mype+1)%npes;		

        int iy_end_top = chunk_size + 1;
        int iy_start_bottom = 0;

        //Set diriclet boundary conditions on left and right boarder
        initialize_boundaries<<<(ny/npes)/128+1,128>>>( a, a_new, PI, iy_start_global-1, nx, chunk_size, ny);
        CUDA_RT_CALL( cudaGetLastError() );
        CUDA_RT_CALL( cudaDeviceSynchronize() );

        CUDA_RT_CALL( cudaStreamCreate(&compute_stream) );
        CUDA_RT_CALL( cudaEventCreateWithFlags ( &compute_done, cudaEventDisableTiming ) );

        CUDA_RT_CALL( cudaMalloc( &l2_norm_d, sizeof(real) ) );
        CUDA_RT_CALL( cudaMallocHost( &l2_norm_h, sizeof(real) ) );

        CUDA_RT_CALL( cudaDeviceSynchronize() );

        if (!mype) { 
            if (!csv) printf("Jacobi relaxation: %d iterations on %d x %d mesh\n", iter_max, ny, nx);
        }

        dim3 dim_block(32,4,1);
        dim3 dim_grid( nx/dim_block.x+1, (ny/npes)/dim_block.y+1, 1 );

        int iter = 0;
        if (!mype) l2_norm = 1.0;
        
        shmem_barrier_all();

        double start = MPI_Wtime();
        PUSH_RANGE("Jacobi solve",0)
        while ( l2_norm > tol && iter < iter_max )
        { 
            CUDA_RT_CALL( cudaMemsetAsync(l2_norm_d, 0 , sizeof(real), compute_stream ) );
            
            jacobi_kernel<<<dim_grid,dim_block,0,compute_stream>>>( a_new, a, l2_norm_d, iy_start, iy_end, nx, top, iy_end_top, bottom, iy_start_bottom );
            CUDA_RT_CALL( cudaGetLastError() );

          
            if ( (iter % nccheck) == 0 || (!csv && (iter % 100) == 0) ) { 
                 CUDA_RT_CALL( cudaMemcpyAsync( l2_norm_h, l2_norm_d, sizeof(real), cudaMemcpyDeviceToHost, compute_stream ) );
            
                 CUDA_RT_CALL( cudaStreamSynchronize( compute_stream ) );

                 MPI_CALL( MPI_Allreduce( l2_norm_h, &l2_norm, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD) );

                l2_norm = std::sqrt( l2_norm );

                if(!csv && (iter % 100) == 0)
                {
                  if (!mype) printf("%5d, %0.6f\n", iter, l2_norm);
                }
            } else { 
                shmem_barrier_all_nb(compute_stream);
                CUDA_RT_CALL( cudaGetLastError() );
	    }

            std::swap(a_new,a);
            iter++;
        }

        shmem_barrier_all();
        double stop = MPI_Wtime();
        POP_RANGE
    
        CUDA_RT_CALL( cudaDeviceSynchronize() ); 
        shmem_barrier_all();

	CUDA_RT_CALL( cudaMemcpy( a_h+iy_start_global*nx, a+nx, chunk_size*nx*sizeof(real), cudaMemcpyDeviceToHost ) );

	result_correct = true;
        for (int iy = iy_start_global; result_correct && (iy <= iy_end_global); ++iy) {
        for (int ix = 1; result_correct && (ix < (nx-1)); ++ix) {
            if ( std::fabs( a_ref_h[ iy * nx + ix ] - a_h[ iy * nx + ix ] ) > tol ) {
                fprintf(stderr,"ERROR on rank %d: a[%d * %d + %d] = %f does not match %f (reference)\n",rank,iy,nx,ix, a_h[ iy * nx + ix ], a_ref_h[ iy * nx + ix ]);
                result_correct = false;
            }
        }}
        
        int global_result_correct = 1;
        MPI_CALL( MPI_Allreduce( &result_correct, &global_result_correct, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD ) );
        result_correct = global_result_correct;
 
        if (!mype && result_correct)
        {
            if (csv) {
                printf( "shmem-comms, %d, %d, %d, %d, %d, 1, %f, %f\n", nx, ny, iter_max, nccheck, npes, (stop-start), runtime_serial );
            }
            else {
                printf( "Num GPUs: %d.\n", npes );
                printf( "%dx%d: 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: %8.2f, efficiency: %8.2f \n", ny,nx, runtime_serial, npes, (stop-start), runtime_serial/(stop-start), runtime_serial/(npes*(stop-start))*100 );
            }
        }

        CUDA_RT_CALL( cudaEventDestroy( compute_done ) );
        CUDA_RT_CALL( cudaStreamDestroy( compute_stream ) );
        
        CUDA_RT_CALL( cudaFreeHost( l2_norm_h ) );
        CUDA_RT_CALL( cudaFree( l2_norm_d ) );
       
	    if (!mype) { 
            CUDA_RT_CALL( cudaFreeHost( a_h ) );
            CUDA_RT_CALL( cudaFreeHost( a_ref_h ) );
	    }

        CUDA_RT_CALL( cudaDeviceReset() );
    }

    shmcleanup(); 
   
    MPI_CALL( MPI_Finalize() ); 

    return ( result_correct == 1 ) ? 0 : 1;
}

double single_gpu(const int nx, const int ny, const int iter_max, real* const a_ref_h, const int nccheck, const bool print, int mype)
{
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
    int iy_end = ny;
    
    a = (real *)shmalloc(nx*(ny+2)*sizeof(real));
    a_new = (real *)shmalloc(nx*(ny+2)*sizeof(real));
    
    CUDA_RT_CALL( cudaMemset( a, 0, nx*(ny+2)*sizeof(real) ) );
    CUDA_RT_CALL( cudaMemset( a_new, 0, nx*(ny+2)*sizeof(real) ) );
    
    //Set diriclet boundary conditions on top and right boarder
    initialize_boundaries<<<ny/128+1,128>>>( a, a_new, PI, 0, nx, ny, ny);

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
    
    if (print) printf("Single GPU jacobi relaxation: %d iterations on %d x %d mesh with norm check every %d iterations\n", iter_max, ny, nx, nccheck);

    dim3 dim_block(32,4,1);
    dim3 dim_grid( nx/dim_block.x+1, ny/dim_block.y+1, 1 );

    int iter = 0;
    real l2_norm = 1.0;
    
    double start = MPI_Wtime();
    PUSH_RANGE("Jacobi solve",0)
    while ( l2_norm > tol && iter < iter_max )
    {
        CUDA_RT_CALL( cudaMemsetAsync(l2_norm_d, 0 , sizeof(real), compute_stream ) );
        
        CUDA_RT_CALL( cudaStreamWaitEvent( compute_stream, push_top_done, 0 ) );
        CUDA_RT_CALL( cudaStreamWaitEvent( compute_stream, push_bottom_done, 0 ) );
        
        jacobi_kernel<<<dim_grid,dim_block,0,compute_stream>>>( a_new, a, l2_norm_d, iy_start, iy_end, nx, mype, iy_end+1, mype, (iy_start-1) );
        CUDA_RT_CALL( cudaGetLastError() );
        CUDA_RT_CALL( cudaEventRecord( compute_done, compute_stream ) );
        
        if ( (iter % nccheck) == 0 || ( print && ( (iter % 100) == 0 ) ) ) {
            CUDA_RT_CALL( cudaMemcpyAsync( l2_norm_h, l2_norm_d, sizeof(real), cudaMemcpyDeviceToHost, compute_stream ) );
        }
        
        if ( (iter % nccheck) == 0 || ( print && ( (iter % 100) == 0 ) ) ) {
            CUDA_RT_CALL( cudaStreamSynchronize( compute_stream ) );
            l2_norm = *l2_norm_h;
            l2_norm = std::sqrt( l2_norm );
            if( print && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, l2_norm);
        }
    
        std::swap(a_new,a);
        iter++;
    }
    POP_RANGE
    double stop = MPI_Wtime();
    
    CUDA_RT_CALL( cudaMemcpy( a_ref_h, a, nx*(ny+2)*sizeof(real), cudaMemcpyDeviceToHost ) );
    
    CUDA_RT_CALL( cudaEventDestroy( push_bottom_done ) );
    CUDA_RT_CALL( cudaEventDestroy( push_top_done ) );
    CUDA_RT_CALL( cudaEventDestroy( compute_done ) );
    CUDA_RT_CALL( cudaStreamDestroy( push_bottom_stream ) );
    CUDA_RT_CALL( cudaStreamDestroy( push_top_stream ) );
    CUDA_RT_CALL( cudaStreamDestroy( compute_stream ) );
    
    CUDA_RT_CALL( cudaFreeHost( l2_norm_h ) );
    CUDA_RT_CALL( cudaFree( l2_norm_d ) );
    
    return (stop-start);
}
