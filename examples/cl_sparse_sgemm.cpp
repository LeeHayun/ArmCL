#ifndef ARM_COMPUTE_CL /* Needed by Utils.cpp to handle OpenCL exceptions properly */
#error "This example needs to be built with -DARM_COMPUTE_CL"
#endif /* ARM_COMPUTE_CL */

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTuner.h"
#include "utils/Utils.h"

#include <cstdlib>

using namespace arm_compute;
using namespace utils;

double measure(CLSparseGEMM *clgemm, int n_times) {
    //arm_compute::CLScheduler::get().default_init();
    //graph->graph_init(true);
    clgemm->run();
    arm_compute::CLScheduler::get().sync();

    auto tbegin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_times; i++) {
        clgemm->run();
    }
    arm_compute::CLScheduler::get().sync();
    auto tend = std::chrono::high_resolution_clock::now();


    double cost = std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count();
    return cost / n_times;
}

class CLSparseSGEMMExample : public Example
{
public:
    void do_setup(int argc, char **argv) override
    {
        NPYLoader npy_m1;
        MTXLoader mtx_m2;
        alpha = 1.0f;
        beta  = 0.0f;

        CLScheduler::get().default_init(&tuner);

        std::ifstream stream;
        if(argc > 1)
        {
            stream.open(argv[1], std::fstream::in);
        }

        if(argc < 3 || (argc < 4 && stream.bad()))
        {
            // Print help
            std::cout << "Usage: 1) ./build/cl_sgemm input_matrix_1.npy input_matrix_2.npy [alpha = 1] [beta = 0]\n";
            std::cout << "       2) ./build/cl_sgemm M N K [alpha = 1.0f] [beta = 0.0f]\n\n";
            std::cout << "Too few or no input_matrices provided. Using M=7, N=3, K=5, alpha=1.0f and beta=0.0f\n\n";

            return;
            /*
            src0.allocator()->init(TensorInfo(TensorShape(5U, 7U), 1, DataType::F32));
            src1_values.allocator()->init(TensorInfo(TensorShape(10U), 1, DataType::F32));
            src1_row_ptr.allocator()->init(TensorInfo(TensorShape(5U+1U), 1, DataType::U32));
            src1_col_idx.allocator()->init(TensorInfo(TensorShape(10U), 1, DataType::U32));
            */
        }
        else
        {
            if(stream.good()) /* case file1.npy file2.mtx [alpha = 1.0f] [beta = 0.0f] */
            {
                npy_m1.open(argv[1]);
                npy_m1.init_tensor(src0, DataType::F32);
                mtx_m2.open(argv[2]);
                mtx_m2.init_tensor(src1_values, src1_row_ptr, src1_col_idx, DataType::F32, MatrixFormat::CSR);

                if(argc > 3)
                {
                    alpha = strtof(argv[3], nullptr);

                    if(argc > 4)
                    {
                        beta = strtof(argv[4], nullptr);
                    }
                }
            }
            //else /* case M N K [alpha = 1.0f] [beta = 0.0f] */
            /*
            {
                size_t M = strtol(argv[1], nullptr, 10);
                size_t N = strtol(argv[2], nullptr, 10);
                size_t K = strtol(argv[3], nullptr, 10);

                src0.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::F32));
                src1_values.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::F32));

                if(argc > 4)
                {
                    alpha = strtof(argv[4], nullptr);

                    if(argc > 5)
                    {
                        beta = strtof(argv[5], nullptr);
                    }
                }
            }
            */
        }
        
        init_sparse_sgemm_output(dst, src0, mtx_m2.columns(), DataType::F32);

        // Configure function
        sparse_csr_gemm.configure(&src0, &src1_values, &src1_col_idx, &src1_row_ptr, nullptr, &dst, alpha, beta);

        // Allocate all the images
        src0.allocator()->allocate();
        src1_values.allocator()->allocate();
        src1_row_ptr.allocator()->allocate();
        src1_col_idx.allocator()->allocate();
        dst.allocator()->allocate();

        // Fill the input images with either the data provided or random data
        if(npy_m1.is_open())
        {
            npy_m1.fill_tensor(src0);
            mtx_m2.fill_tensor(src1_values, src1_row_ptr, src1_col_idx);
            print_tensor<float>(&src0);
            print_tensor<float>(&src1_values);
            print_tensor<unsigned int>(&src1_row_ptr);
            print_tensor<unsigned int>(&src1_col_idx);

            output_filename = "sgemm_out.npy";
            is_fortran      = npy_m1.is_fortran();
        }
        else
        {
            ARM_COMPUTE_ERROR("Sparse format doesn't support fill_random_tensor() yet");
            return;
            /*
            fill_random_tensor(src0, -1.f, 1.f);
            fill_random_tensor(src1_values, -1.f, 1.f);
            fill_random_tensor(src1_values, -1.f, 1.f);
            fill_random_tensor(src1_values, -1.f, 1.f);
            */
        }

        // Dummy run for CLTuner
        std::cout << "Demmy run for CLTuner" << std::endl;
        sparse_csr_gemm.run();
        tuner.save_to_file("tune.csv");
    }
    void do_run() override
    {
        // Execute the function
        std::cout << "Execute the function" << std::endl;
        sparse_csr_gemm.run();

        // Make sure all the OpenCL jobs are done executing:
        CLScheduler::get().sync();

        measure(&sparse_csr_gemm, 50);
        double tt = measure(&sparse_csr_gemm, 100);
        std::cout << "Time: " << tt << std::endl;

        print_tensor<float>(&dst);
    }
    void do_teardown() override
    {
        if(!output_filename.empty()) /* Save to .npy file */
        {
            save_to_npy(dst, output_filename, is_fortran);
        }
    }

    template <typename T>
    void print_tensor(CLTensor *tensor, bool print_coord = false)
    {
        //tensor.print(std::cout, IOFormatInfo(IOFormatInfo::PrintRegion::Full));
        return;
        
        tensor->map(true);

        Window window;
        T val;
        window.use_tensor_dimensions(tensor->info()->tensor_shape());
        Iterator it(tensor, window);
        execute_window_loop(window, [&](const Coordinates & id)
        {
            if (id.x() == 0)
                std::cout << std::endl;
            if (print_coord)
                std::cout << "(" << id.x() << ", " << id.y() << ", " << id.z() << ") ";

            val = *reinterpret_cast<T *>(it.ptr());
            std::cout << val << " ";
        },
        it);
        std::cout << std::endl;

        tensor->unmap();
    }

private:
    CLTensor     src0{}, src1_values{}, src1_row_ptr{}, src1_col_idx{}, dst{};
    CLSparseGEMM sparse_csr_gemm{};
    CLTuner      tuner{};
    float        alpha{}, beta{};
    bool         is_fortran{};
    std::string  output_filename{};
};

/** Main program for sgemm test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Matrix A, [optional] Matrix B, [optional] alpha, [optional] beta )
 */
int main(int argc, char **argv)
{
    return utils::run_example<CLSparseSGEMMExample>(argc, argv);
}
