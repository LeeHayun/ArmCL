#ifndef __CISS_SUPPORT_H__
#define __CISS_SUPPORT_H__

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/Utility.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/ITensorAccessor.h"
#include "arm_compute/graph/Types.h"
#include "arm_compute/runtime/Tensor.h"
//#include "opencv2/opencv.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/core/core.hpp"
#include "utils/CommonGraphOptions.h"

//#include <array>
//#include <random>
//#include <string>
//#include <vector>

namespace arm_compute
{
namespace graph_utils
{
//Gwang
class GetOutputAccessor final : public graph::ITensorAccessor
{
public:
    float *_dst_data{};
    GetOutputAccessor(float *dst_data, std::string _filename);
    GetOutputAccessor(GetOutputAccessor &&) = default;

    bool access_tensor(ITensor &tensor) override;

private:
    const std::string _filename;
};

inline std::unique_ptr<graph::ITensorAccessor> get_output_ciss(const arm_compute::utils::CommonGraphParams &graph_parameters, float *dst_data)
{
    return arm_compute::support::cpp14::make_unique<GetOutputAccessor>(dst_data, graph_parameters.image);
}

} // namespace graph_utils
} // namespace arm_compute

#endif /* __CISS_SUPPORT_H__ */
