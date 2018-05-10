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
#include <unistd.h>
#include <mpi.h>

#define CALC_KERNEL_TIME

#define MPI_CALL( call ) \
{   \
    int mpi_status = call; \
    if ( 0 != mpi_status ) \
    { \
        char mpi_error_string[MPI_MAX_ERROR_STRING]; \
        int mpi_error_string_length = 0; \
        printf("MIAOOO\n"); \
        MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
        if ( NULL != mpi_error_string ) \
            fprintf(stderr, "ERROR: MPI call \"%s\" in line %d of file %s failed with %s (%d).\n", #call, __LINE__, __FILE__, mpi_error_string, mpi_status); \
        else \
            fprintf(stderr, "ERROR: MPI call \"%s\" in line %d of file %s failed with %d.\n", #call, __LINE__, __FILE__, mpi_status); \
    } \
}

#include <cuda_runtime.h>

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
#define PUSH_RANGE(name,cid) do { } while(0)
#define POP_RANGE()  do { } while(0)
#endif

#define CUDA_RT_CALL( call )                                            \
    {                                                                   \
        cudaError_t cudaStatus = call;                                  \
        if ( cudaSuccess != cudaStatus ) {                              \
            fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n", \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    }

#include <comm.h>

#ifdef USE_DOUBLE
    typedef double real;
    #define MPI_REAL_TYPE MPI_DOUBLE
#else
    typedef float real;
    #define MPI_REAL_TYPE MPI_FLOAT
#endif

constexpr real tol = 1.0e-8;

const real PI  = 2.0 * std::asin(1.0);

void launch_dummy_kernel(cudaStream_t stream);

void launch_initialize_boundaries(
    real* __restrict__ const a_new,
    real* __restrict__ const a,
    const real pi,
    const int offset,
    const int nx, const int my_ny, const int ny );

void launch_jacobi_kernel(
          real* __restrict__ const a_new,
    const real* __restrict__ const a,
          real* __restrict__ const l2_norm,
    const int iy_start, const int iy_end,
    const int nx,
    cudaStream_t stream);

void launch_comm_kernel(comm_dev_descs_t descs, cudaStream_t stream, int num_ranks);

void launch_jacobi_comm_kernel(
    real* __restrict__ const a_new,
    const real* __restrict__ const a,
    real* __restrict__ const l2_norm,
    const int iy_start, const int iy_end,
    const int nx,
    cudaStream_t stream,
    comm_dev_descs_t descs);


std::pair<double,float> single_gpu(const int nx, const int ny, const int iter_max, real* const a_ref_h, const int nccheck, const bool print);

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
    MPI_CALL( MPI_Init(&argc,&argv) );
    int rank;
    MPI_CALL( MPI_Comm_rank(MPI_COMM_WORLD,&rank) );
    int size;
    MPI_CALL( MPI_Comm_size(MPI_COMM_WORLD,&size) );
    
    const int iter_max = get_argval<int>(argv, argv+argc,"-niter", 1000);
    const int nccheck = get_argval<int>(argv, argv+argc,"-nccheck", 1);
    const int nx = get_argval<int>(argv, argv+argc,"-nx", 1024);
    const int ny = get_argval<int>(argv, argv+argc,"-ny", 1024);
    const bool csv = get_arg(argv, argv+argc,"-csv");

    // Ensure correctness if ny%size != 0
    int chunk_size = std::ceil( (1.0*(ny-2))/size );
    
    int selected_gpu = 0;
    char * value = getenv("USE_GPU"); 
    if (value != NULL) {
        selected_gpu = atoi(value);
    } else {
        int local_rank = -1;
        MPI_Comm local_comm;
        MPI_CALL( MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm) );
        
        MPI_CALL( MPI_Comm_rank(local_comm,&local_rank) );
        
        MPI_CALL( MPI_Comm_free(&local_comm) );
        selected_gpu = local_rank;
    }
    
    int n_gpus;
    CUDA_RT_CALL( cudaGetDeviceCount(&n_gpus) );
    for (int gpu = 0; gpu < n_gpus; ++gpu) {
        cudaDeviceProp prop;
        CUDA_RT_CALL( cudaGetDeviceProperties(&prop, gpu) );
        printf("[%d] GPU%d name=%s clockRate=%d memoryClockRate=%d multiProcessorCount=%d %s\n",
               rank, gpu, prop.name, prop.clockRate, prop.memoryClockRate, prop.multiProcessorCount,
               gpu == selected_gpu ? "<==" : "");
    }

    CUDA_RT_CALL( cudaSetDevice( selected_gpu ) ); 
    CUDA_RT_CALL( cudaFree( 0 ) );
    
    real* a_ref_h;
    CUDA_RT_CALL( cudaMallocHost( &a_ref_h, nx*ny*sizeof(real) ) );
    real* a_h;
    CUDA_RT_CALL( cudaMallocHost( &a_h, nx*ny*sizeof(real) ) );
    double runtime_serial = 0;
    float kernel_serial = 0;
    std::tie(runtime_serial,kernel_serial) = single_gpu(nx,ny,iter_max,a_ref_h,nccheck,!csv&&(0==rank));

    real* a;
    CUDA_RT_CALL( cudaMalloc( &a, nx*(chunk_size+2)*sizeof(real) ) );
    real* a_new;
    int use_gpu_buffers=0;

    if(use_gpu_buffers == 1)
    {
        CUDA_RT_CALL( cudaMalloc( &a_new, nx*(chunk_size+2)*sizeof(real) ) );
        CUDA_RT_CALL( cudaMemset( a_new, 0, nx*(chunk_size+2)*sizeof(real) ) );        
    }
    else
    {
        CUDA_RT_CALL( cudaMallocHost( &a_new, nx*(chunk_size+2)*sizeof(real) ) );
        memset( a_new, 0, nx*(chunk_size+2)*sizeof(real) );
    }

    printf("[%d] allocated a/a_new size=%d reals\n", rank, nx*(chunk_size+2)); usleep(1);
    CUDA_RT_CALL( cudaMemset( a, 0, nx*(chunk_size+2)*sizeof(real) ) );
    CUDA_RT_CALL( cudaMemset( a_new, 0, nx*(chunk_size+2)*sizeof(real) ) );

    comm_reg_t a_reg = nullptr;
    comm_reg_t a_new_reg = nullptr;
    if (comm_use_comm()) {
        printf("[%d] using libmp/comm in %s mode\n", rank, comm_use_model_sa()?"async":"sync");
        COMM_CHECK(comm_init(MPI_COMM_WORLD, rank));
        printf("registering a=%p\n", a);
        COMM_CHECK(comm_register(a,     nx*(chunk_size+2), MPI_REAL_TYPE, &a_reg));
        printf("registering a_new=%p\n", a_new);
        COMM_CHECK(comm_register(a_new, nx*(chunk_size+2), MPI_REAL_TYPE, &a_new_reg));
    } else {
        printf("[%d] using MPI\n", rank);
    }
    //Calculate local domain boundaries
    int iy_start_global = rank * chunk_size + 1;            // 1      | cs+1
    int iy_end_global = iy_start_global + chunk_size - 1;   // 1+cs-1 | cs+1+cs-1
    // Do not process boundaries
    iy_end_global = std::min( iy_end_global, ny - 2 );
    
    int iy_start = 1;                                           // 1      
    int iy_end = (iy_end_global-iy_start_global + 1)+iy_start;  // cs-1+1+1=cs+1

    //Set diriclet boundary conditions on left and right boarder
    launch_initialize_boundaries( a, a_new, PI, iy_start_global-1, nx, (chunk_size+2), ny );
    CUDA_RT_CALL( cudaDeviceSynchronize() );

    cudaStream_t compute_stream;
    CUDA_RT_CALL( cudaStreamCreate(&compute_stream) );
    cudaEvent_t compute_done;
    CUDA_RT_CALL( cudaEventCreateWithFlags ( &compute_done, cudaEventDisableTiming ) );
   
    real* l2_norm_d;
    CUDA_RT_CALL( cudaMalloc( &l2_norm_d, sizeof(real) ) );
    real* l2_norm_h;
    CUDA_RT_CALL( cudaMallocHost( &l2_norm_h, sizeof(real) ) );

    PUSH_RANGE("MPI_Warmup",5);
    for (int i=0; i<10;++i) {
        const int top = rank > 0 ? rank - 1 : (size-1);
        const int bottom = (rank+1)%size;

        if (comm_use_comm()) {
            comm_request_t ready_req[2];
            comm_request_t recv_req[2];
            comm_request_t send_req[2];
            
            // Async - KI Model
            if (comm_use_model_ki()) {
                COMM_CHECK(comm_irecv(a_new+(iy_end  *nx), nx, MPI_REAL_TYPE, &a_new_reg, bottom, &recv_req[0]));
                COMM_CHECK(comm_send_ready_on_stream(bottom, &ready_req[0], compute_stream));
                COMM_CHECK(comm_irecv(a_new, nx, MPI_REAL_TYPE, &a_new_reg, top, &recv_req[1]));
                COMM_CHECK(comm_send_ready_on_stream(top, &ready_req[1], compute_stream));
                
                COMM_CHECK(comm_prepare_wait_ready(top));
                COMM_CHECK(comm_prepare_isend(a_new+(iy_start*nx), nx, MPI_REAL_TYPE, &a_new_reg, top, &send_req[0]));
                COMM_CHECK(comm_prepare_wait_ready(bottom));
                COMM_CHECK(comm_prepare_isend(a_new+(iy_end-1)*nx, nx, MPI_REAL_TYPE, &a_new_reg, bottom, &send_req[1]));

                COMM_CHECK(comm_prepare_wait_all(2, recv_req));
                
                comm_dev_descs_t descs = comm_prepared_requests();
                launch_comm_kernel(descs, compute_stream, size);

                COMM_CHECK(comm_wait_all_on_stream(2, send_req, compute_stream));
                COMM_CHECK(comm_wait_all_on_stream(2, ready_req, compute_stream));
            }
            else if (comm_use_model_sa()) {
                COMM_CHECK(comm_irecv(a_new+(iy_end  *nx), nx, MPI_REAL_TYPE, &a_new_reg, bottom, &recv_req[0]));
                COMM_CHECK(comm_send_ready_on_stream(bottom, &ready_req[0], compute_stream));
                COMM_CHECK(comm_irecv(a_new, nx, MPI_REAL_TYPE, &a_new_reg, top, &recv_req[1]));
                COMM_CHECK(comm_send_ready_on_stream(top, &ready_req[1], compute_stream));

                COMM_CHECK(comm_wait_ready_on_stream(top, compute_stream));
                COMM_CHECK(comm_isend_on_stream(a_new+(iy_start*nx), nx, MPI_REAL_TYPE, &a_new_reg, top,    &send_req[0], compute_stream));

                COMM_CHECK(comm_wait_ready_on_stream(bottom, compute_stream));
                COMM_CHECK(comm_isend_on_stream(a_new+(iy_end-1)*nx, nx, MPI_REAL_TYPE, &a_new_reg, bottom, &send_req[1], compute_stream));


                COMM_CHECK(comm_wait_all_on_stream(2, recv_req, compute_stream));
                COMM_CHECK(comm_wait_all_on_stream(2, send_req, compute_stream));
                COMM_CHECK(comm_wait_all_on_stream(2, ready_req, compute_stream));

                COMM_CHECK(comm_flush());
            } else {
                COMM_CHECK(comm_irecv(a_new+(iy_end*nx), nx, MPI_REAL_TYPE, &a_new_reg, bottom, &recv_req[0]));
                COMM_CHECK(comm_send_ready(bottom, &ready_req[0]));
                COMM_CHECK(comm_wait_ready(top));
                COMM_CHECK(comm_isend(a_new+(iy_start*nx), nx, MPI_REAL_TYPE, &a_new_reg, top,    &send_req[0]));
                COMM_CHECK(comm_wait_all(1, &recv_req[0]));
                COMM_CHECK(comm_wait_all(1, &send_req[0]));

                COMM_CHECK(comm_irecv(a_new,             nx, MPI_REAL_TYPE, &a_new_reg, top,    &recv_req[1]));
                COMM_CHECK(comm_send_ready(top,    &ready_req[1]));
                COMM_CHECK(comm_wait_ready(bottom));
                COMM_CHECK(comm_isend(a_new+(iy_end-1)*nx, nx, MPI_REAL_TYPE, &a_new_reg, bottom, &send_req[1]));
                COMM_CHECK(comm_wait_all(1, &recv_req[1]));
                COMM_CHECK(comm_wait_all(1, &send_req[1]));

                //COMM_CHECK(comm_flush());
            }
            std::swap(a_new_reg,a_reg);
        } else {
            MPI_CALL( MPI_Sendrecv( a_new+iy_start*nx,   nx, MPI_REAL_TYPE, top   , 0, a_new+(iy_end*nx), nx, MPI_REAL_TYPE, bottom, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE ));
            MPI_CALL( MPI_Sendrecv( a_new+(iy_end-1)*nx, nx, MPI_REAL_TYPE, bottom, 0, a_new,             nx, MPI_REAL_TYPE, top,    0, MPI_COMM_WORLD, MPI_STATUS_IGNORE ));
        }

        std::swap(a_new,a);
    }
    POP_RANGE();

    launch_dummy_kernel(compute_stream);
    CUDA_RT_CALL( cudaDeviceSynchronize() );

    if (!csv && 0 == rank)
    {
        printf("Jacobi relaxation: %d iterations on %d x %d mesh with norm check every %d iterations\n", iter_max, ny, nx, nccheck);
    }

    int iter = 0;
    real l2_norm = 1.0;

    //printf("[%d] before Jacobi\n", rank);
    //sleep(1);
    MPI_CALL( MPI_Barrier(MPI_COMM_WORLD) );
    double start = MPI_Wtime();
    PUSH_RANGE("Jacobi solve",0);
    while ( l2_norm > tol && iter < iter_max )
    {
        //printf("[%d] before kernel\n", rank);
        CUDA_RT_CALL( cudaMemsetAsync(l2_norm_d, 0 , sizeof(real), compute_stream ) );

        if(!comm_use_model_ki())
        {
            launch_jacobi_kernel( a_new, a, l2_norm_d, iy_start, iy_end, nx, compute_stream );
            CUDA_RT_CALL( cudaEventRecord( compute_done, compute_stream ) );
         
            if ( (iter % nccheck) == 0 || (!csv && (iter % 100) == 0) ) {
                CUDA_RT_CALL( cudaMemcpyAsync( l2_norm_h, l2_norm_d, sizeof(real), cudaMemcpyDeviceToHost, compute_stream ) );
            }
        }

        const int top = rank > 0 ? rank - 1 : (size-1);
        const int bottom = (rank+1)%size;
        
        //Apply periodic boundary conditions

        PUSH_RANGE("MPI",5);

        //printf("[%d] i=%d before comms\n", rank, iter);

        if (comm_use_comm()) {
            comm_request_t ready_req[2];
            comm_request_t recv_req[2];
            comm_request_t send_req[2];
            if (comm_use_model_ki()) {
                COMM_CHECK(comm_irecv(a_new+(iy_end  *nx), nx, MPI_REAL_TYPE, &a_new_reg, bottom, &recv_req[0]));
                COMM_CHECK(comm_send_ready_on_stream(bottom, &ready_req[0], compute_stream));
                COMM_CHECK(comm_irecv(a_new, nx, MPI_REAL_TYPE, &a_new_reg, top, &recv_req[1]));
                COMM_CHECK(comm_send_ready_on_stream(top, &ready_req[1], compute_stream));
                
                COMM_CHECK(comm_prepare_wait_ready(top));
                COMM_CHECK(comm_prepare_isend(a_new+(iy_start*nx), nx, MPI_REAL_TYPE, &a_new_reg, top, &send_req[0]));
                COMM_CHECK(comm_prepare_wait_ready(bottom));
                COMM_CHECK(comm_prepare_isend(a_new+(iy_end-1)*nx, nx, MPI_REAL_TYPE, &a_new_reg, bottom, &send_req[1]));

                COMM_CHECK(comm_prepare_wait_all(2, recv_req));
                
                // ----------- Compute and Communicate
                comm_dev_descs_t descs = comm_prepared_requests();
                launch_jacobi_comm_kernel( a_new, a, l2_norm_d, iy_start, iy_end, nx, compute_stream, descs, ordered_sends );
                if ( (iter % nccheck) == 0 || (!csv && (iter % 100) == 0) ) {
                    CUDA_RT_CALL( cudaMemcpyAsync( l2_norm_h, l2_norm_d, sizeof(real), cudaMemcpyDeviceToHost, compute_stream ) );
                }
                // -----------------------------------
                COMM_CHECK(comm_wait_all_on_stream(2, send_req, compute_stream));
                COMM_CHECK(comm_wait_all_on_stream(2, ready_req, compute_stream));
            }
            else if (comm_use_model_sa()) {
                COMM_CHECK(comm_irecv(a_new+(iy_end  *nx), nx, MPI_REAL_TYPE, &a_new_reg, bottom, &recv_req[0]));
                COMM_CHECK(comm_send_ready_on_stream(bottom, &ready_req[0], compute_stream));
                COMM_CHECK(comm_irecv(a_new, nx, MPI_REAL_TYPE, &a_new_reg, top, &recv_req[1]));
                COMM_CHECK(comm_send_ready_on_stream(top, &ready_req[1], compute_stream));

                COMM_CHECK(comm_wait_ready_on_stream(top, compute_stream));
                COMM_CHECK(comm_isend_on_stream(a_new+(iy_start*nx), nx, MPI_REAL_TYPE, &a_new_reg, top,    &send_req[0], compute_stream));

                COMM_CHECK(comm_wait_ready_on_stream(bottom, compute_stream));
                COMM_CHECK(comm_isend_on_stream(a_new+(iy_end-1)*nx, nx, MPI_REAL_TYPE, &a_new_reg, bottom, &send_req[1], compute_stream));


                COMM_CHECK(comm_wait_all_on_stream(2, recv_req, compute_stream));
                COMM_CHECK(comm_wait_all_on_stream(2, send_req, compute_stream));
                COMM_CHECK(comm_wait_all_on_stream(2, ready_req, compute_stream));
                // moving flush out of loop seems to be detrimental
                //COMM_CHECK(comm_flush());
            } else {
                CUDA_RT_CALL( cudaEventSynchronize( compute_done ) );
                COMM_CHECK(comm_irecv(a_new+(iy_end*nx), nx, MPI_REAL_TYPE, &a_new_reg, bottom, &recv_req[0]));
                COMM_CHECK(comm_send_ready(bottom, &ready_req[0]));
                COMM_CHECK(comm_irecv(a_new,             nx, MPI_REAL_TYPE, &a_new_reg, top,    &recv_req[1]));
                COMM_CHECK(comm_send_ready(top,    &ready_req[1]));

                COMM_CHECK(comm_wait_ready(top));
                COMM_CHECK(comm_isend(a_new+(iy_start*nx), nx, MPI_REAL_TYPE, &a_new_reg, top,    &send_req[0]));
                COMM_CHECK(comm_wait_ready(bottom));
                COMM_CHECK(comm_isend(a_new+(iy_end-1)*nx, nx, MPI_REAL_TYPE, &a_new_reg, bottom, &send_req[1]));

                COMM_CHECK(comm_wait_all(2, recv_req));
                COMM_CHECK(comm_wait_all(2, send_req));

                //COMM_CHECK(comm_flush());
            }
            std::swap(a_new_reg,a_reg);
        } else {
            CUDA_RT_CALL( cudaEventSynchronize( compute_done ) );
            //CUDA_RT_CALL( cudaStreamSynchronize( compute_stream ) );
            MPI_CALL( MPI_Sendrecv( a_new+iy_start*nx,   nx, MPI_REAL_TYPE, top   , 0, a_new+(iy_end*nx), nx, MPI_REAL_TYPE, bottom, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE ));
            MPI_CALL( MPI_Sendrecv( a_new+(iy_end-1)*nx, nx, MPI_REAL_TYPE, bottom, 0, a_new,             nx, MPI_REAL_TYPE, top,    0, MPI_COMM_WORLD, MPI_STATUS_IGNORE ));
        }

        POP_RANGE();

        if ( (iter % nccheck) == 0 || (!csv && (iter % 100) == 0) ) {
            if (comm_use_comm() && comm_use_model_sa())
                COMM_CHECK(comm_flush());
            CUDA_RT_CALL( cudaStreamSynchronize( compute_stream ) );
            MPI_CALL( MPI_Allreduce( l2_norm_h, &l2_norm, 1, MPI_REAL_TYPE, MPI_SUM, MPI_COMM_WORLD ) );
            l2_norm = std::sqrt( l2_norm );

            if(!csv && 0 == rank && (iter % 100) == 0)
            {
                printf("%5d, %0.6f\n", iter, l2_norm);
            }
        }

        std::swap(a_new,a);
        iter++;
    }

    if (comm_use_comm() && comm_use_model_sa())
        COMM_CHECK(comm_flush());
    CUDA_RT_CALL( cudaStreamSynchronize( compute_stream ) );

    double stop = MPI_Wtime();
    POP_RANGE();

    CUDA_RT_CALL( cudaMemcpy( a_h+iy_start_global*nx, a+nx, std::min((ny-iy_start_global)*nx,chunk_size*nx)*sizeof(real), cudaMemcpyDeviceToHost ) );

    int result_correct = 1;
    for (int iy = iy_start_global; result_correct && (iy < iy_end_global); ++iy) {
    for (int ix = 1; result_correct && (ix < (nx-1)); ++ix) {
        if ( std::fabs( a_ref_h[ iy * nx + ix ] - a_h[ iy * nx + ix ] ) > tol ) {
            fprintf(stderr,"ERROR on rank %d: a[%d * %d + %d] = %f does not match %f (reference)\n",rank,iy,nx,ix, a_h[ iy * nx + ix ], a_ref_h[ iy * nx + ix ]);
            result_correct = 0;
        }
    }}

    int global_result_correct = 1;
    MPI_CALL( MPI_Allreduce( &result_correct, &global_result_correct, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD ) );
    result_correct = global_result_correct;

    if (rank == 0 && result_correct)
    {
        if (csv) {
            printf( "mpi, %d, %d, %d, %d, %d, 1, %f, %f\n", nx, ny, iter_max, nccheck, size, (stop-start), runtime_serial );
        }
        else {
            printf( "Num GPUs: %d.\n", size );
            printf( "%dx%d: 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: %8.2f, efficiency: %8.2f \n", ny,nx, runtime_serial, size, (stop-start), runtime_serial/(stop-start), runtime_serial/(size*(stop-start))*100 );
            printf( "1 GPU: single kernel execution took %8.6f s\n", kernel_serial);
        }
    }
    CUDA_RT_CALL( cudaEventDestroy( compute_done ) );
    CUDA_RT_CALL( cudaStreamDestroy( compute_stream ) );
    
    CUDA_RT_CALL( cudaFreeHost( l2_norm_h ) );
    CUDA_RT_CALL( cudaFree( l2_norm_d ) );
    
    if (use_gpu_buffers == 1)
    { CUDA_RT_CALL( cudaFree( a_new ) ); }
    else
    { CUDA_RT_CALL( cudaFreeHost( a_new ) ); }

    CUDA_RT_CALL( cudaFree( a ) );
    
    CUDA_RT_CALL( cudaFreeHost( a_h ) );
    CUDA_RT_CALL( cudaFreeHost( a_ref_h ) );
    
    MPI_CALL( MPI_Finalize() );
    return ( result_correct == 1 ) ? 0 : 1;
}

std::pair<double,float> single_gpu(const int nx, const int ny, const int iter_max, real* const a_ref_h, const int nccheck, const bool print)
{
    real* a;
    real* a_new;
    
    cudaStream_t compute_stream;
    cudaStream_t push_top_stream;
    cudaStream_t push_bottom_stream;
#ifdef CALC_KERNEL_TIME
    cudaEvent_t kernel_start;
#endif
    cudaEvent_t compute_done;
    cudaEvent_t push_top_done;
    cudaEvent_t push_bottom_done;
    
    real* l2_norm_d;
    real* l2_norm_h;
    
    int iy_start = 1;
    int iy_end = (ny-1);
    
    CUDA_RT_CALL( cudaMalloc( &a, nx*ny*sizeof(real) ) );
    CUDA_RT_CALL( cudaMalloc( &a_new, nx*ny*sizeof(real) ) );
    
    CUDA_RT_CALL( cudaMemset( a, 0, nx*ny*sizeof(real) ) );
    CUDA_RT_CALL( cudaMemset( a_new, 0, nx*ny*sizeof(real) ) );
    
    //Set diriclet boundary conditions on left and right boarder
    launch_initialize_boundaries( a, a_new, PI, 0, nx, ny, ny );
    CUDA_RT_CALL( cudaDeviceSynchronize() );
    
    CUDA_RT_CALL( cudaStreamCreate(&compute_stream) );
    CUDA_RT_CALL( cudaStreamCreate(&push_top_stream) );
    CUDA_RT_CALL( cudaStreamCreate(&push_bottom_stream) );
#ifdef CALC_KERNEL_TIME
    CUDA_RT_CALL( cudaEventCreateWithFlags ( &kernel_start, cudaEventDefault ) );
    CUDA_RT_CALL( cudaEventCreateWithFlags ( &compute_done, cudaEventDefault ) );
#else
    CUDA_RT_CALL( cudaEventCreateWithFlags ( &compute_done, cudaEventDisableTiming ) );
#endif
    CUDA_RT_CALL( cudaEventCreateWithFlags ( &push_top_done, cudaEventDisableTiming ) );
    CUDA_RT_CALL( cudaEventCreateWithFlags ( &push_bottom_done, cudaEventDisableTiming ) );
    
    CUDA_RT_CALL( cudaMalloc( &l2_norm_d, sizeof(real) ) );
    CUDA_RT_CALL( cudaMallocHost( &l2_norm_h, sizeof(real) ) );
    
    CUDA_RT_CALL( cudaDeviceSynchronize() );
    
    if (print) printf("Single GPU jacobi relaxation: %d iterations on %d x %d mesh with norm check every %d iterations\n", iter_max, ny, nx, nccheck);

    int iter = 0;
    real l2_norm = 1.0;
    float last_kernel_ms = 0.0f;
    
    double start = MPI_Wtime();
    PUSH_RANGE("Jacobi solve",0);
    while ( l2_norm > tol && iter < iter_max )
    {
        CUDA_RT_CALL( cudaMemsetAsync(l2_norm_d, 0 , sizeof(real), compute_stream ) );
        
        CUDA_RT_CALL( cudaStreamWaitEvent( compute_stream, push_top_done, 0 ) );
        CUDA_RT_CALL( cudaStreamWaitEvent( compute_stream, push_bottom_done, 0 ) );
#ifdef CALC_KERNEL_TIME        
        CUDA_RT_CALL( cudaEventRecord( kernel_start, compute_stream ) );
#endif
        launch_jacobi_kernel( a_new, a, l2_norm_d, iy_start, iy_end, nx, compute_stream );
        CUDA_RT_CALL( cudaEventRecord( compute_done, compute_stream ) );
        
        if ( (iter % nccheck) == 0 || (iter % 100) == 0 ) {
            CUDA_RT_CALL( cudaMemcpyAsync( l2_norm_h, l2_norm_d, sizeof(real), cudaMemcpyDeviceToHost, compute_stream ) );
        }
        
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
            nx*sizeof(real), cudaMemcpyDeviceToDevice, push_bottom_stream ) );
        CUDA_RT_CALL( cudaEventRecord( push_bottom_done, push_bottom_stream ) );
        
        if ( (iter % nccheck) == 0 || (iter % 100) == 0 ) {
            CUDA_RT_CALL( cudaStreamSynchronize( compute_stream ) );
#ifdef CALC_KERNEL_TIME
            CUDA_RT_CALL( cudaEventElapsedTime( &last_kernel_ms, kernel_start, compute_done ) );
#endif
            l2_norm = *l2_norm_h;
            l2_norm = std::sqrt( l2_norm );
            if(print && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, l2_norm);
        }
        
        std::swap(a_new,a);
        iter++;
    }
    CUDA_RT_CALL( cudaStreamSynchronize( compute_stream ) );
    POP_RANGE();
    double stop = MPI_Wtime();
#ifdef CALC_KERNEL_TIME
    CUDA_RT_CALL( cudaEventElapsedTime( &last_kernel_ms, kernel_start, compute_done ) );
#else
    last_kernel_ms = .0f;
#endif
    
    CUDA_RT_CALL( cudaMemcpy( a_ref_h, a, nx*ny*sizeof(real), cudaMemcpyDeviceToHost ) );
    
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
    return std::make_pair(stop-start, last_kernel_ms/1000.0);
}
