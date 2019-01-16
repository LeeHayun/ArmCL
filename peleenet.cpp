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

/** Example demonstrating how to implement ShuffleNet network using the Compute Library's graph API */
class PeleeNetExample : public Example
{
public:
    PeleeNetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "PeleeNet")
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

        // Set default layout if needed (Single kernel grouped convolution not yet supported int NHWC)
        if(!common_opts.data_layout->is_set())
        {
            common_params.data_layout = DataLayout::NCHW;
        }

        // Checks
        ARM_COMPUTE_EXIT_ON_MSG(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type), "QASYMM8 not supported for this graph");
        ARM_COMPUTE_EXIT_ON_MSG(common_params.data_type == DataType::F16 && common_params.target == Target::NEON, "F16 NEON not supported for this graph");

        // Print parameter values
        std::cout << common_params << std::endl;
        std::cout << "Model: Peleenet" << std::endl;

        // Create model path
        std::string model_path = "/cnn_data/peleenet_model/";

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Add model path to data path
        if(!data_path.empty())
        {
            data_path += model_path;
        }

        // Create input descriptor
        const TensorShape tensor_shape     = permute_shape(TensorShape(224U, 224U, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;

        // Create preprocessor
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<TFPreproccessor>(0);

        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor), false /* Do not convert to BGR */));

        unsigned int num_init_features = 32;
        unsigned int growth_rate = 32;

        add_stem_block(weights_layout, num_init_features);

        add_dense_blocks(weights_layout, 3, growth_rate, 1);
        add_transition_layer(weights_layout, 128, true);

        add_dense_blocks(weights_layout, 4, growth_rate, 2);
        add_transition_layer(weights_layout, 256, true);

        add_dense_blocks(weights_layout, 8, growth_rate, 4);
        add_transition_layer(weights_layout, 512, true);

        add_dense_blocks(weights_layout, 6, growth_rate, 4);
        add_transition_layer(weights_layout, 704, false);

        graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG)).set_name("predictions/AvgPool")
              << FlattenLayer().set_name("predictions/Reshape")
              << FullyConnectedLayer(
                  1000U,
                  get_weights_accessor(data_path, "pred_w_0.npy", weights_layout),
                  get_weights_accessor(data_path, "pred_b_0.npy"))
              .set_name("predictions/FC")
              << SoftmaxLayer().set_name("predictions/Softmax")
              << OutputLayer(get_output_accessor(common_params, 5));

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
        graph.run();
        graph.run();
        graph.run();
        graph.run();
        finish = getTime();
        printf("First run: %lf sec\n", finish-start);

        start = getTime();
        graph.run();
        graph.run();
        graph.run();
        graph.run();
        graph.run();
        graph.run();
        graph.run();
        graph.run();
        graph.run();
        graph.run();
        finish = getTime();
        printf("Second run: %lf sec\n", finish-start);
    }

private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;

    void add_stem_block(DataLayout weights_layout, unsigned int num_init_features)
    {
        graph << ConvolutionLayer(
                  3U, 3U, num_init_features,
                  get_random_accessor(1.f, 1.f),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(2, 2, 1, 1))
              << BatchNormalizationLayer(
                  get_random_accessor(1.f, 1.f),
                  get_random_accessor(1.f, 1.f),
                  get_random_accessor(1.f, 1.f),
                  get_random_accessor(1.f, 1.f),
                  0.001f)
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream left_ss(graph);
        SubStream right_ss(graph);

        left_ss << ConvolutionLayer(
                    1U, 1U, num_init_features / 2,
                    get_random_accessor(1.f, 1.f),
                    std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                    PadStrideInfo(1, 1, 0, 0))
                << BatchNormalizationLayer(
                    get_random_accessor(1.f, 1.f),
                    get_random_accessor(1.f, 1.f),
                    get_random_accessor(1.f, 1.f),
                    get_random_accessor(1.f, 1.f),
                    0.001f)
                << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                << ConvolutionLayer(
                    3U, 3U, num_init_features,
                    get_random_accessor(1.f, 1.f),
                    std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                    PadStrideInfo(2, 2, 1, 1))
                << BatchNormalizationLayer(
                    get_random_accessor(1.f, 1.f),
                    get_random_accessor(1.f, 1.f),
                    get_random_accessor(1.f, 1.f),
                    get_random_accessor(1.f, 1.f),
                    0.001f)
                << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        right_ss << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)));

        graph << ConcatLayer(std::move(left_ss), std::move(right_ss))
              << ConvolutionLayer(
                  3U, 3U, num_init_features,
                  get_random_accessor(1.f, 1.f),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(1, 1, 1, 1))
              << BatchNormalizationLayer(
                  get_random_accessor(1.f, 1.f),
                  get_random_accessor(1.f, 1.f),
                  get_random_accessor(1.f, 1.f),
                  get_random_accessor(1.f, 1.f),
                  0.001f)
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
    }

    void add_dense_blocks(DataLayout weights_layout, unsigned int num_block, unsigned int growth_rate, unsigned int bottleneck_width)
    {
        unsigned int k = growth_rate / 2;
        unsigned int inter_channel = (unsigned int)(k * bottleneck_width / 4) * 4;

        for (int i = 0; i < num_block; i++)
        {
            SubStream left_ss(graph);
            SubStream right_ss(graph);
            SubStream id_ss(graph);

            left_ss << ConvolutionLayer(
                        1U, 1U, inter_channel,
                        get_random_accessor(1.f, 1.f),
                        std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                        PadStrideInfo(1, 1, 0, 0))
                    << BatchNormalizationLayer(
                        get_random_accessor(1.f, 1.f),
                        get_random_accessor(1.f, 1.f),
                        get_random_accessor(1.f, 1.f),
                        get_random_accessor(1.f, 1.f),
                        0.001f)
                    << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                    << ConvolutionLayer(
                        3U, 3U, k,
                        get_random_accessor(1.f, 1.f),
                        std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                        PadStrideInfo(1, 1, 1, 1))
                    << BatchNormalizationLayer(
                        get_random_accessor(1.f, 1.f),
                        get_random_accessor(1.f, 1.f),
                        get_random_accessor(1.f, 1.f),
                        get_random_accessor(1.f, 1.f),
                        0.001f)
                    << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

            right_ss << ConvolutionLayer(
                         1U, 1U, inter_channel,
                         get_random_accessor(1.f, 1.f),
                         std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                         PadStrideInfo(1, 1, 0, 0))
                     << BatchNormalizationLayer(
                         get_random_accessor(1.f, 1.f),
                         get_random_accessor(1.f, 1.f),
                         get_random_accessor(1.f, 1.f),
                         get_random_accessor(1.f, 1.f),
                         0.001f)
                     << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                     << ConvolutionLayer(
                         3U, 3U, k,
                         get_random_accessor(1.f, 1.f),
                         std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                         PadStrideInfo(1, 1, 1, 1))
                     << BatchNormalizationLayer(
                         get_random_accessor(1.f, 1.f),
                         get_random_accessor(1.f, 1.f),
                         get_random_accessor(1.f, 1.f),
                         get_random_accessor(1.f, 1.f),
                         0.001f)
                     << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                     << ConvolutionLayer(
                         3U, 3U, k,
                         get_random_accessor(1.f, 1.f),
                         std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                         PadStrideInfo(1, 1, 1, 1))
                     << BatchNormalizationLayer(
                         get_random_accessor(1.f, 1.f),
                         get_random_accessor(1.f, 1.f),
                         get_random_accessor(1.f, 1.f),
                         get_random_accessor(1.f, 1.f),
                         0.001f)
                     << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

            graph << ConcatLayer(std::move(id_ss), std::move(left_ss), std::move(right_ss));
        }
    }

    void add_transition_layer(DataLayout weights_layout, unsigned int output_channel, bool is_avgpool)
    {
        graph << ConvolutionLayer(
                  1U, 1U, output_channel,
                  get_random_accessor(1.f, 1.f),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(1, 1, 0, 0))
              << BatchNormalizationLayer(
                  get_random_accessor(1.f, 1.f),
                  get_random_accessor(1.f, 1.f),
                  get_random_accessor(1.f, 1.f),
                  get_random_accessor(1.f, 1.f),
                  0.001f)
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        if (is_avgpool)
        {
            graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 2, PadStrideInfo(2, 2, 0, 0)));
        }
    }
};

/** Main program for ShuffleNet
 *
 * Model is based on:
 *      https://arxiv.org/abs/1707.01083
 *      "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices"
 *      Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<PeleeNetExample>(argc, argv);
}
