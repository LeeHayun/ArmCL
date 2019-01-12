/*
 * Copyright (c) 2018 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/graph.h"
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

#include "ciss_support.h"

#include <sys/time.h>
#include <unistd.h>

double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)1.0e-6*tv.tv_usec;
}

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

/** Example demonstrating how to implement YOLOv3 network using the Compute Library's graph API */
class GraphCISSExample : public Example
{
public:
    GraphCISSExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "CISS")
    {
    }

    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Checks
        ARM_COMPUTE_EXIT_ON_MSG(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type), "QASYMM8 not supported for this graph");

        // Print parameter values
        std::cout << common_params << std::endl;

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Create a preprocessor object
        const std::array<float, 3> mean_rgb{ {123.68f, 116.779f, 103.939f} };
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb);

        // Create input descriptor
        const TensorShape tensor_shape     = permute_shape(TensorShape(1248U, 384U, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;

        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor)))
              << ConvolutionLayer(
                3U, 3U, 64U,
                get_weights_accessor(data_path, "conv1_kernels:0.npy", weights_layout), 
                get_weights_accessor(data_path, "conv1_biases:0.npy"), 
                PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        SubStream conv1(graph);
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)));
        graph << get_fire_node(data_path, "fire2", weights_layout, 16U, 64U, 64U);
        graph << get_fire_node(data_path, "fire3", weights_layout, 16U, 64U, 64U);
        SubStream fire3(graph);
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)));
        graph << get_fire_node(data_path, "fire4", weights_layout, 32U, 128U, 128U);
        graph << get_fire_node(data_path, "fire5", weights_layout, 32U, 128U, 128U);
        SubStream fire5(graph);
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)));
        graph << get_fire_node(data_path, "fire6", weights_layout, 48U, 192U, 192U);
        graph << get_fire_node(data_path, "fire7", weights_layout, 48U, 192U, 192U);
        graph << get_fire_node(data_path, "fire8", weights_layout, 64U, 256U, 256U);
        graph << get_fire_node(data_path, "fire9", weights_layout, 64U, 256U, 256U);

        SubStream fire9(graph);

        graph << get_fire_node(data_path, "fire10", weights_layout, 96U, 384U, 384U);
        graph << get_fire_node(data_path, "fire11", weights_layout, 96U, 384U, 384U);
        graph << ConvolutionLayer(
                3U, 3U, 72U,
                get_weights_accessor(data_path, "conv12_kernels:0.npy", weights_layout), 
                get_weights_accessor(data_path, "conv12_biases:0.npy"), 
                PadStrideInfo(1 ,1, 1, 1))
              << OutputLayer(get_output_accessor(common_params, 5));
              //<< OutputLayer(get_output_ciss(common_params, dst_data));

        fire9 << get_fire_deconv_node(fire9, data_path, "fire_deconv10", weights_layout, 64U, 128U, 128U);
        graph << EltwiseLayer(std::move(fire5), std::move(fire9), EltwiseOperation::Add);
        SubStream fire10(graph);

        fire10 << get_fire_deconv_node(fire10, data_path, "fire_deconv11", weights_layout, 32U, 64U, 64U);
        graph << EltwiseLayer(std::move(fire3), std::move(fire10), EltwiseOperation::Add);
        SubStream fire11(graph);

        fire11 << get_fire_deconv_node(fire11, data_path, "fire_deconv12", weights_layout, 16U, 32U, 32U);
        graph << EltwiseLayer(std::move(conv1), std::move(fire11), EltwiseOperation::Add);
        SubStream fire12(graph);

        fire12 << get_fire_deconv_node(fire12, data_path, "fire_deconv13", weights_layout, 8U, 16U, 16U);
        fire12 << ConvolutionLayer(
                  3U, 3U, 2U,
                  get_weights_accessor(data_path, "conv14_kernels:0.npy", weights_layout),
                  get_weights_accessor(data_path, "conv14_biases:0.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << OutputLayer(get_output_accessor(common_params, 5));
              //<< OutputLayer(get_output_ciss(common_params, dst_data));



        // Finalize graph
        GraphConfig config;
        config.num_threads = common_params.threads;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_file  = common_params.tuner_file;

        graph.finalize(common_params.target, config);

        return true;
    }
    void do_run() override
    {
        double start, finish;
        // Run graph
        start = getTime();
        graph.run();
        finish = getTime();
        printf("First run: %lf sec\n", finish-start);

        start = getTime();
        graph.run();
        finish = getTime();
        printf("Second run: %lf sec\n", finish-start);
    }

private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;

    float *dst_data{};

    ConcatLayer get_fire_node(const std::string &data_path, std::string &&param_path, DataLayout weights_layout, unsigned int s1x1, unsigned int e1x1, unsigned int e3x3)
    {
        graph << ConvolutionLayer(
                  1U, 1U, s1x1,
                  get_weights_accessor(data_path, param_path + "_" + "squeeze1x1_kernels:0.npy", weights_layout), 
                  get_weights_accessor(data_path, param_path + "_" + "squeeze1x1_biases:0.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream expand1x1(graph);
        expand1x1 << ConvolutionLayer(
                      1U, 1U, e1x1,
                      get_weights_accessor(data_path, param_path + "_" + "expand1x1_kernels:0.npy", weights_layout),
                      get_weights_accessor(data_path, param_path + "_" + "expand1x1_biases:0.npy"),
                      PadStrideInfo(1, 1, 0, 0))
                  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream expand3x3(graph);
        expand3x3 << ConvolutionLayer(
                      3U, 3U, e3x3,
                      get_weights_accessor(data_path, param_path + "_" + "expand3x3_kernels:0.npy", weights_layout),
                      get_weights_accessor(data_path, param_path + "_" + "expand3x3_biases:0.npy"),
                      PadStrideInfo(1, 1, 1, 1))
                  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return ConcatLayer(std::move(expand1x1), std::move(expand3x3));
    }

    ConcatLayer get_fire_deconv_node(SubStream &sub_graph, const std::string &data_path, std::string &&param_path, DataLayout weights_layout, unsigned int s1x1, unsigned int e1x1, unsigned int e3x3)
    {
        sub_graph << ConvolutionLayer(
                  1U, 1U, s1x1,
                  get_weights_accessor(data_path, param_path + "_" + "squeeze1x1_kernels:0.npy", weights_layout),
                  get_weights_accessor(data_path, param_path + "_" + "squeeze1x1_biases:0.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << DeconvolutionLayer(
                  4U, 4U, s1x1,
                  get_weights_accessor(data_path, param_path + "_" + "deconv_kernels:0.npy", weights_layout),
                  get_weights_accessor(data_path, param_path + "_" + "deconv_biases:0.npy"),
                  PadStrideInfo(2, 2, 1, 1),
                  Size2D(0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream expand1x1(sub_graph);
        expand1x1 << ConvolutionLayer(
                      1U, 1U, e1x1,
                      get_weights_accessor(data_path, param_path + "_" + "expand1x1_kernels:0.npy", weights_layout),
                      get_weights_accessor(data_path, param_path + "_" + "expand1x1_biases:0.npy"),
                      PadStrideInfo(1, 1, 0, 0))
                  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream expand3x3(sub_graph);
        expand3x3 << ConvolutionLayer(
                      3U, 3U, e3x3,
                      get_weights_accessor(data_path, param_path + "_" + "expand3x3_kernels:0.npy", weights_layout),
                      get_weights_accessor(data_path, param_path + "_" + "expand3x3_biases:0.npy"),
                      PadStrideInfo(1, 1, 1, 1))
                  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return ConcatLayer(std::move(expand1x1), std::move(expand3x3));
    }

};

/** Main program for YOLOv3
 *
 * Model is based on:
 *      https://arxiv.org/abs/1804.02767
 *      "YOLOv3: An Incremental Improvement"
 *      Joseph Redmon, Ali Farhadi
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 *
 * @return Return code
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphCISSExample>(argc, argv);
}
