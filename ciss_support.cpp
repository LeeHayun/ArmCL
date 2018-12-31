#include "ciss_support.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/runtime/SubTensor.h"
#include "utils/ImageLoader.h"
#include "utils/Utils.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

#include <iomanip>
#include <limits>

using namespace cv;
using namespace arm_compute::graph_utils;

typedef struct{
    float x, y, w, h;
} box;

typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;

detection *make_network_boxes(float thresh, int *num)
{
    int num_classes = 3;
    int i;
    int nboxes = 78*24*9; // threshold
    if(num) *num = nboxes;
    detection *dets = (detection*)calloc(nboxes, sizeof(detection));
    for(i = 0; i < nboxes; ++i){
        dets[i].prob = (float*)calloc(num_classes, sizeof(float));
        /*
        if(l.coords > 4){
            dets[i].mask = calloc(l.coords-4, sizeof(float));
        }
        */
    }
    return dets;
}

float anchor_shapes[9][2] = {
    {  36.,  37.}, { 366., 174.}, { 115.,  59.},
    { 162.,  87.}, {  38.,  90.}, { 258., 173.},
    { 224., 108.}, {  78., 170.}, {  72.,  43.}
};

void get_detections(float *predictions, int w, int h, float thresh, detection *dets)
{
    int i,j,n;
    int n_classes = 3;
    int side_w = 78;
    int side_h = 24;
    int n_anchors = 9;
    float *_predictions = predictions;
    //int per_cell = 5*num+classes;
    for (i = 0; i < side_w*side_h; ++i){
        int col = i % side_w;
        int row = i / side_w;

        int class_index = i*n_anchors*(n_classes+1+4); //TODO: depth (+1)
        int conf_index = class_index + n_anchors*n_classes;
        int delta_index = conf_index + n_anchors;
        //int depth_index = delta_index + n_anchors*4;

        for(n = 0; n < n_anchors; ++n){
            int index = i*n_anchors + n;
            //int p_index = 78*24*n_classes + i*9 + n;
            float conf = 1. / (1. + exp(-_predictions[conf_index + n]));
            //int box_index = l.side*l.side*(l.classes + l.n) + (i*l.n + n)*4;
            box b;

            b.x = predictions[delta_index + n*4 + 0];
            b.y = predictions[delta_index + n*4 + 1];
            b.w = predictions[delta_index + n*4 + 2];
            b.h = predictions[delta_index + n*4 + 3];

            b.x = ((col+1) * 1248. / (side_w + 1.)) + b.x * anchor_shapes[n][0];
            b.y = ((row+1) * 384. / (side_h + 1.)) + b.y * anchor_shapes[n][1];
            b.w = anchor_shapes[n][0] * ((b.w >= 1.0) ? (exp(1.0)*b.w) : exp(b.w));
            b.h = anchor_shapes[n][1] * ((b.h >= 1.0) ? (exp(1.0)*b.h) : exp(b.h));

            dets[index].bbox = b;

            dets[index].objectness = conf;

            float sum_class = 0.;
            for(j = 0; j < n_classes; ++j){
                sum_class += exp(_predictions[class_index+j]);
            }

            for(j = 0; j < n_classes; ++j){
                float prob = conf*exp(_predictions[class_index+j])/sum_class;
                dets[index].prob[j] = (prob > thresh) ? prob : 0;
                //if (dets[index].prob[j] > 0) printf("%f\n", dets[index].prob[j]);
            }
        }
    }
}

detection *get_network_boxes(float *predictions, int w, int h, float thresh, int *num)
{
    detection *dets = make_network_boxes(thresh, num);
    get_detections(predictions, w, h, thresh, dets);
    return dets;
}

int nms_comparator(const void *pa, const void *pb)
{
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if(b.sort_class >= 0){
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

void do_nms_sort(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total-1;
    for(i = 0; i <= k; ++i){
        if(dets[i].objectness == 0){
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k+1;

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator);
        for(i = 0; i < total; ++i){
            if(dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for(j = i+1; j < total; ++j){
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

void free_detections(detection *dets, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(dets[i].prob);
        if(dets[i].mask) free(dets[i].mask);
    }
    free(dets);
}

char class_name[3][16] = {"car", "pedestrian", "cyclist"};

void vis_detections(cv::Mat& cv_image, detection *dets, int num, float thresh, int classes)
{
    int fontFace = cv::FONT_HERSHEY_PLAIN;
    double fontScale = 2/2.;
    double thickness = 1.5/2.;
    int baseline = 0;

    cv::Size s = cv_image.size();
    std::cout << "height: " << s.height << ", width: " << s.width << std::endl;

    for(int i = 0; i < num; i++) {
        int _class = -1;
        for(int j = 0; j < classes; j++) {
            if (dets[i].prob[j] > thresh) {
                if (_class < 0) {
                    _class = j;
                }
            }
        }
        if (_class >= 0) {
            int width = 1248 * .006;

            box b = dets[i].bbox;
std::cout << "(" << b.x << ", " << b.y << ", " << b.w << ", " << b.h << "): " << class_name[_class] << " / " << dets[i].prob[_class] << std::endl;
            int left = (b.x - b.w/2.);
            int right = (b.x + b.w/2.);
            int top = (b.y + b.h/2.);
            int bot = (b.y - b.h/2.);

            if (left < 0) left = 0;
            if (right > 1248-1) right = 1248-1;
            if (bot < 0) bot = 0;
            if (top > 384-1) top = 384-1;

            cv::rectangle(cv_image, cv::Point(left,bot),
                          cv::Point(right,top),
                          cv::Scalar(0, 0, 255), 2, 8, 0);
            string text = std::string(class_name[_class]) + " " + std::to_string(dets[i].prob[_class]);
            cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
            cv::rectangle(cv_image, cv::Point(left,bot - 2),
                          cv::Point(left + textSize.width/1.3, bot - 2 - textSize.height),
                          cv::Scalar::all(180), CV_FILLED);
            cv::putText(cv_image, text, cv::Point(left,bot - 2),
                        fontFace, thickness, cv::Scalar::all(0), fontScale, 8);
        }
    }

    /*
    cv::imshow("Detections", cv_image);
    cv::waitKey(0);
    */
}


//#define TEST

GetOutputAccessor::GetOutputAccessor(float *dst_data, const std::string filename)
    : _filename(filename)
{
      _dst_data = dst_data;
}

bool GetOutputAccessor::access_tensor(ITensor &tensor)
{
    ARM_COMPUTE_UNUSED(tensor);

    size_t CHANNELS = tensor.info()->dimension(0);
    size_t GRID_W = tensor.info()->dimension(1);
    size_t GRID_H = tensor.info()->dimension(2);
    size_t NUM_CLASS = 3;
    size_t ANCHORS_PER_GRID = 9;
    
    //72 CLASS, CONFIDENCE, Coordinate
    std::cout << tensor.info()->dimension(0) << std::endl;
    //78 GRID_W
    std::cout << tensor.info()->dimension(1) << std::endl;
    //24 GRID_H
    std::cout << tensor.info()->dimension(2) << std::endl;
    //1
    std::cout << tensor.info()->dimension(3) << std::endl;
    //shape
    std::cout << tensor.info()->tensor_shape() << std::endl;
    
    std::cout << "----------------------------------" << std::endl;
    
     
    std::cout << "element_size : " << tensor.info()->element_size() << std::endl;

    Window output_window;
#ifdef TEST
    output_window.use_tensor_dimensions(tensor.info()->tensor_shape());
#else
    output_window.use_tensor_dimensions(tensor.info()->tensor_shape(), Window::DimY);
#endif
    std::cout << " Dimensions of the output's iterator:\n";
    std::cout << " X = [start=" << output_window.x().start() << ", end=" << output_window.x().end() << ", step=" << output_window.x().step() << "]\n";
    std::cout << " Y = [start=" << output_window.y().start() << ", end=" << output_window.y().end() << ", step=" << output_window.y().step() << "]\n";
    std::cout << " Z = [start=" << output_window.z().start() << ", end=" << output_window.z().end() << ", step=" << output_window.z().step() << "]\n";

    Iterator output_it(&tensor, output_window);

    _dst_data = new float[GRID_W * GRID_H * CHANNELS];

    execute_window_loop(output_window, [&](const Coordinates & id)
            {
                //std::cout << "Copying one row starting from [" << id.x() << "," << id.y() << "," << id.z() << "]\n";
#ifdef TEST
                float val = *reinterpret_cast<float *>(output_it.ptr());
                std::cout << "(" << id.z() << "," << id.y() << "," << id.x() << "): " << *reinterpret_cast<float *>(output_it.ptr()) << "\n";
#else
                // Copy one whole row:
                memcpy(_dst_data + id.z() * (tensor.info()->dimension(1)*tensor.info()->dimension(0)) + id.y() * tensor.info()->dimension(0), output_it.ptr(), tensor.info()->dimension(0) * sizeof(float));
#endif
            }, output_it);

#ifndef TEST

    int nboxes = 0;
    detection *dets = get_network_boxes(_dst_data, 1248, 384, 0.005, &nboxes);
    do_nms_sort(dets, nboxes, NUM_CLASS, 0.6);

    Mat image;
    
    image = imread(_filename, IMREAD_COLOR);


    vis_detections(image, dets, nboxes, 0.7, NUM_CLASS);
    free_detections(dets, nboxes);
  
    //Rect r = Rect(10,20,50,50);
    
    //rectangle(image, r, Scalar(255,0,0), 1, 8, 0);
    
    imwrite("/home/odroid/result.jpg", image);

#endif
    
    return 0;
}
