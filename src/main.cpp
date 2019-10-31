// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine object_detection demo application
* \file object_detection_demo_yolov3_async/main.cpp
* \example object_detection_demo_yolov3_async/main.cpp
*/
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

#include "object_detection_demo_yolov3_async.hpp"

#include <ext_list.hpp>

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cmath>


using namespace std;
using namespace cv;


using namespace InferenceEngine;

#define yolo_scale_13 13
#define yolo_scale_26 26
#define yolo_scale_52 52





bool Sobel(const Mat& image, Mat& result, int TYPE)
{
	if (image.channels() != 1)
		return false;
	// 系数设置
	int kx(0);
	int ky(0);
	if (TYPE == 0) {
		kx = 0; ky = 1;
	}
	else if (TYPE == 1) {
		kx = 1; ky = 0;
	}
	else if (TYPE == 2) {
		kx = 1; ky = 1;
	}
	else
		return false;

	// 设置mask
	float mask[3][3] = { {1,2,1},{0,0,0},{-1,-2,-1} };
	Mat y_mask = Mat(3, 3, CV_32F, mask) / 8;
	Mat x_mask = y_mask.t(); // 转置

	// 计算x方向和y方向上的滤波
	Mat sobelX, sobelY;
	filter2D(image, sobelX, CV_32F, x_mask);
	filter2D(image, sobelY, CV_32F, y_mask);
	sobelX = abs(sobelX);
	sobelY = abs(sobelY);
	// 梯度图
	Mat gradient = kx * sobelX.mul(sobelX) + ky * sobelY.mul(sobelY);

	// 计算阈值
	int scale = 4;
	double cutoff = scale * mean(gradient)[0];

	result.create(image.size(), image.type());
	result.setTo(0);
	for (int i = 1; i < image.rows - 1; i++)
	{
		float* sbxPtr = sobelX.ptr<float>(i);
		float* sbyPtr = sobelY.ptr<float>(i);
		float* prePtr = gradient.ptr<float>(i - 1);
		float* curPtr = gradient.ptr<float>(i);
		float* lstPtr = gradient.ptr<float>(i + 1);
		uchar* rstPtr = result.ptr<uchar>(i);
		// 阈值化和极大值抑制
		for (int j = 1; j < image.cols - 1; j++)
		{
			if (curPtr[j] > cutoff && (
				(sbxPtr[j] > kx*sbyPtr[j] && curPtr[j] > curPtr[j - 1] && curPtr[j] > curPtr[j + 1]) ||
				(sbyPtr[j] > ky*sbxPtr[j] && curPtr[j] > prePtr[j] && curPtr[j] > lstPtr[j])))
				rstPtr[j] = 255;
		}
	}

	return true;
}



void warp_image(Mat src,Mat dst_warp,int choose)//透视变换
{
	Point2f srcPoints[4];//原图中的四点 ,一个包含三维点（x，y）的数组，其中x、y是浮点型数
	Point2f dstPoints[4];//目标图中的四点  


	srcPoints[0] = Point2f(0.117*src.cols, src.rows);
	srcPoints[1] = Point2f(0.422*src.cols,0.6666* src.rows);
	srcPoints[2] = Point2f(0.578*src.cols, 0.6666 * src.rows);
	srcPoints[3] = Point2f(0.883*src.cols, src.rows);			  
	//映射后的四个坐标值
	dstPoints[0] = Point2f(src.cols*0.25, src.rows);
	dstPoints[1] = Point2f(src.cols*0.25, 0);
	dstPoints[2] = Point2f(0.75*src.cols, 0);
	dstPoints[3] = Point2f(0.75*src.cols, src.rows);

	//变形矩阵
	Mat M1 = getPerspectiveTransform(srcPoints, dstPoints);//由四个点对计算透视变换矩阵  
	Mat M2= getPerspectiveTransform(dstPoints, srcPoints);
	if(choose==0)
	{
	warpPerspective(src, dst_warp, M1, src.size(), INTER_LINEAR);//仿射变换  
	}
	if (choose == 1)
	{
		warpPerspective(src, dst_warp, M2, src.size(), INTER_LINEAR);//仿射变换  
	}
}


bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
{
	//Number of key points
	int N = key_point.size();

	//构造矩阵X
	cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < n + 1; j++)
		{
			for (int k = 0; k < N; k++)
			{
				X.at<double>(i, j) = X.at<double>(i, j) +
					std::pow(key_point[k].x, i + j);
			}
		}
	}

	//构造矩阵Y
	cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int k = 0; k < N; k++)
		{
			Y.at<double>(i, 0) = Y.at<double>(i, 0) +
				std::pow(key_point[k].x, i) * key_point[k].y;
		}
	}

	A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	//求解矩阵A
	cv::solve(X, Y, A, cv::DECOMP_LU);
	return true;
}


bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validating the input arguments--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }
    return true;
}

void FrameToBlob(const cv::Mat &frame, InferRequest::Ptr &inferRequest, const std::string &inputName) {
    if (FLAGS_auto_resize) {
        /* Just set input blob containing read image. Resize and layout conversion will be done automatically */
        inferRequest->SetBlob(inputName, wrapMat2Blob(frame));
    } else {
        /* Resize and copy data from the image to the input blob */
        Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
        matU8ToBlob<uint8_t>(frame, frameBlob);
    }
}

static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

struct DetectionObject {
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) {
        this->xmin = static_cast<int>((x - w / 2) * w_scale);
        this->ymin = static_cast<int>((y - h / 2) * h_scale);
        this->xmax = static_cast<int>(this->xmin + w * w_scale);
        this->ymax = static_cast<int>(this->ymin + h * h_scale);
        this->class_id = class_id;
        this->confidence = confidence;
    }

    bool operator<(const DetectionObject &s2) const {
        return this->confidence < s2.confidence;
    }
};

double IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2) {
    double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
    double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
    double area_of_overlap;
    if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
        area_of_overlap = 0;
    else
        area_of_overlap = width_of_overlap_area * height_of_overlap_area;
    double box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin);
    double box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin);
    double area_of_union = box_1_area + box_2_area - area_of_overlap;
    return area_of_overlap / area_of_union;
}

void ParseYOLOV3Output(const CNNLayerPtr &layer, const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold, std::vector<DetectionObject> &objects) {
    // --------------------------- Validating output parameters -------------------------------------
    if (layer->type != "RegionYolo")
        throw std::runtime_error("Invalid output type: " + layer->type + ". RegionYolo expected");
    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    if (out_blob_h != out_blob_w)
        throw std::runtime_error("Invalid size of output " + layer->name +
        " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
        ", current W = " + std::to_string(out_blob_h));
    // --------------------------- Extracting layer parameters -------------------------------------
    auto num = layer->GetParamAsInt("num");
    try { num = layer->GetParamAsInts("mask").size(); } catch (...) {}
    auto coords = layer->GetParamAsInt("coords");
    auto classes = layer->GetParamAsInt("classes");
    std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};
    try { anchors = layer->GetParamAsFloats("anchors"); } catch (...) {}
    auto side = out_blob_h;
    int anchor_offset = 0;

    //throw std::runtime_error("anchors.size() ==" + std::to_string(anchors.size()));

    if (anchors.size() == 18) {        // YoloV3
        switch (side) {
            case yolo_scale_13:
                anchor_offset = 2 * 6;
                break;
            case yolo_scale_26:
                anchor_offset = 2 * 3;
                break;
            case yolo_scale_52:
                anchor_offset = 2 * 0;
                break;
            default:
                throw std::runtime_error("Invalid output size");
        }
    } else if (anchors.size() == 12) { // tiny-YoloV3
        switch (side) {
            case yolo_scale_13:
                anchor_offset = 2 * 3;
                break;
            case yolo_scale_26:
                anchor_offset = 2 * 0;
                break;
            default:
                throw std::runtime_error("Invalid output size");
        }
    } else {                           // unknow anchor
        switch (side) {
            case yolo_scale_13:
                anchor_offset = 2 * 6;
                break;
            case yolo_scale_26:
                anchor_offset = 2 * 3;
                break;
            case yolo_scale_52:
                anchor_offset = 2 * 0;
                break;
            default:
                throw std::runtime_error("Invalid output size");
        }
    }
    auto side_square = side * side;
    const float *output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i) {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < num; ++n) {
            int obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords);
            int box_index = EntryIndex(side, coords, classes, n * side * side + i, 0);
            float scale = output_blob[obj_index];
            if (scale < threshold)
                continue;
            double x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
            double y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
            double height = std::exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1];
            double width = std::exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n];
            for (int j = 0; j < classes; ++j) {
                int class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j);
                float prob = scale * output_blob[class_index];
                if (prob < threshold)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob,
                        static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                        static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }
        }
    }
}


int main(int argc, char *argv[]) {
    try {
        /** This demo covers a certain topology and cannot be generalized for any object detection **/
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validating the input arguments ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        slog::info << "Reading input" << slog::endl;
        cv::VideoCapture cap;
        if (FLAGS_i == "cam0") {
            cap.open(0);
        } else if (FLAGS_i == "cam1") {
            cap.open(1);
        } else if (FLAGS_i == "cam2") {
            cap.open(2);
        } else if (!(cap.open(FLAGS_i.c_str()))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }

        // read input (video) frame
        cv::Mat frame;  cap >> frame;
        cv::Mat next_frame;
       
        const size_t width  = (size_t) cap.get(cv::CAP_PROP_FRAME_WIDTH);
        const size_t height = (size_t) cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        if (!cap.grab()) {
            throw std::logic_error("This demo supports only video (or camera) inputs !!! "
                                   "Failed to get next frame from the " + FLAGS_i);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load Plugin for inference engine -------------------------------------
        slog::info << "Loading plugin" << slog::endl;
        InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(FLAGS_d);
        printPluginVersion(plugin, std::cout);

        /**Loading extensions to the plugin **/

        /** Loading default extensions **/
        if (FLAGS_d.find("CPU") != std::string::npos) {
            /**
             * cpu_extensions library is compiled from the "extension" folder containing
             * custom CPU layer implementations.
            **/
            plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
        }

        if (!FLAGS_l.empty()) {
            // CPU extensions are loaded as a shared library and passed as a pointer to the base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l.c_str());
            plugin.AddExtension(extension_ptr);
        }
        if (!FLAGS_c.empty()) {
            // GPU extensions are loaded from an .xml description and OpenCL kernel files
            plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
        }

        /** Per-layer metrics **/
        if (FLAGS_pc) {
            plugin.SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) ------------
        slog::info << "Loading network files" << slog::endl;
        CNNNetReader netReader;
        /** Reading network model **/
        netReader.ReadNetwork(FLAGS_m);
        /** Setting batch size to 1 **/
        slog::info << "Batch size is forced to  1." << slog::endl;
        netReader.getNetwork().setBatchSize(1);
        /** Extracting the model name and loading its weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netReader.ReadWeights(binFileName);
        /** Reading labels (if specified) **/
        std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";
        std::vector<std::string> labels;
        std::ifstream inputFile(labelFileName);
        std::copy(std::istream_iterator<std::string>(inputFile),
                  std::istream_iterator<std::string>(),
                  std::back_inserter(labels));
        // -----------------------------------------------------------------------------------------------------

        /** YOLOV3-based network should have one input and three output **/
        // --------------------------- 3. Configuring input and output -----------------------------------------
        // --------------------------------- Preparing input blobs ---------------------------------------------
        slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("This demo accepts networks that have only one input");
        }
        InputInfo::Ptr& input = inputInfo.begin()->second;
        auto inputName = inputInfo.begin()->first;
        input->setPrecision(Precision::U8);
        if (FLAGS_auto_resize) {
            input->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
            input->getInputData()->setLayout(Layout::NHWC);
        } else {
            input->getInputData()->setLayout(Layout::NCHW);
        }
        // --------------------------------- Preparing output blobs -------------------------------------------
        slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        //if (outputInfo.size() != 3) {
        //    throw std::logic_error("This demo only accepts networks with three layers");
        //}
        for (auto &output : outputInfo) {
            output.second->setPrecision(Precision::FP32);
            output.second->setLayout(Layout::NCHW);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the plugin ------------------------------------------
        slog::info << "Loading model to the plugin" << slog::endl;
        ExecutableNetwork network = plugin.LoadNetwork(netReader.getNetwork(), {});

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Creating infer request -----------------------------------------------
        InferRequest::Ptr async_infer_request_next = network.CreateInferRequestPtr();
        InferRequest::Ptr async_infer_request_curr = network.CreateInferRequestPtr();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Doing inference ------------------------------------------------------
        slog::info << "Start inference " << slog::endl;

        bool isLastFrame = false;
        bool isAsyncMode = false;  // execution is always started using SYNC mode
        bool isModeChanged = false;  // set to TRUE when execution mode is changed (SYNC<->ASYNC)

        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        auto total_t0 = std::chrono::high_resolution_clock::now();
        auto wallclock = std::chrono::high_resolution_clock::now();
        double ocv_decode_time = 0, ocv_render_time = 0;

        while (true) {
            auto t0 = std::chrono::high_resolution_clock::now();
            // Here is the first asynchronous point:
            // in the Async mode, we capture frame to populate the NEXT infer request
            // in the regular mode, we capture frame to the CURRENT infer request
            if (!cap.read(next_frame)) {
                if (next_frame.empty()) {
                    isLastFrame = true;  // end of video file
                } else {
                    throw std::logic_error("Failed to get frame from cv::VideoCapture");
                }
            }
            if (isAsyncMode) {
                if (isModeChanged) {
                    FrameToBlob(frame, async_infer_request_curr, inputName);
                }
                if (!isLastFrame) {
                    FrameToBlob(next_frame, async_infer_request_next, inputName);
                }
            } else if (!isModeChanged) {
                FrameToBlob(frame, async_infer_request_curr, inputName);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            ocv_decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            t0 = std::chrono::high_resolution_clock::now();
            // Main sync point:
            // in the true Async mode, we start the NEXT infer request while waiting for the CURRENT to complete
            // in the regular mode, we start the CURRENT request and wait for its completion
            if (isAsyncMode) {
                if (isModeChanged) {
                    async_infer_request_curr->StartAsync();
                }
                if (!isLastFrame) {
                    async_infer_request_next->StartAsync();
                }
            } else if (!isModeChanged) {
                async_infer_request_curr->StartAsync();
            }

            if (OK == async_infer_request_curr->Wait(IInferRequest::WaitMode::RESULT_READY)) {
                t1 = std::chrono::high_resolution_clock::now();
                ms detection = std::chrono::duration_cast<ms>(t1 - t0);

                t0 = std::chrono::high_resolution_clock::now();
                ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
                wallclock = t0;

                t0 = std::chrono::high_resolution_clock::now();
                std::ostringstream out;
                out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
                    << (ocv_decode_time + ocv_render_time) << " ms";
                cv::putText(frame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                out.str("");
               // out << "Wallclock time " << (isAsyncMode ? "(TRUE ASYNC):      " : "(SYNC, press Tab): ");
               /// out << std::fixed << std::setprecision(2) << wall.count() << " ms (" << 1000.f / wall.count() << " fps)";
                cv::putText(frame, out.str(), cv::Point2f(0, 50), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                if (!isAsyncMode) {  // In the true async mode, there is no way to measure detection time directly
                    out.str("");
                    out << "Detection time  : " << std::fixed << std::setprecision(2) << detection.count()
                        << " ms ("
                        << 1000.f / detection.count() << " fps)";
                    cv::putText(frame, out.str(), cv::Point2f(0, 75), 	       cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
                }

                // ---------------------------Processing output blobs--------------------------------------------------
                // Processing results of the CURRENT request
                unsigned long resized_im_h = inputInfo.begin()->second.get()->getDims()[0];
                unsigned long resized_im_w = inputInfo.begin()->second.get()->getDims()[1];
                std::vector<DetectionObject> objects;
                // Parsing outputs
                for (auto &output : outputInfo) {
                    auto output_name = output.first;
                    //slog::info << "output_name = " + output_name << slog::endl;
                    CNNLayerPtr layer = netReader.getNetwork().getLayerByName(output_name.c_str());
                    Blob::Ptr blob = async_infer_request_curr->GetBlob(output_name);
                    ParseYOLOV3Output(layer, blob, resized_im_h, resized_im_w, height, width, FLAGS_t, objects);
                }
                // Filtering overlapping boxes
                std::sort(objects.begin(), objects.end());
                for (std::size_t i = 0; i < objects.size(); ++i) {
                    if (objects[i].confidence == 0)
                        continue;
                    for (std::size_t j = i + 1; j < objects.size(); ++j) {
                        if (IntersectionOverUnion(objects[i], objects[j]) >= FLAGS_iou_t) {
                            objects[j].confidence = 0;
                        }
                        //if (objects[j].confidence == 1) {
                        //    objects[j].confidence = 0;
                        //}
                    }
                }
                // Drawing boxes
                for (auto &object : objects) {
                    if (object.confidence < FLAGS_t)
                        continue;
                    auto label = object.class_id;
                    float confidence = object.confidence;
                    if (FLAGS_r) {
                        std::cout << "[" << label << "] element, prob = " << confidence <<
                                  "    (" << object.xmin << "," << object.ymin << ")-(" << object.xmax << "," << object.ymax << ")"
                                  << ((confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << std::endl;
                    }
                    //slog::info << "confidence = " + std::to_string(confidence) << slog::endl;
                    if (confidence > FLAGS_t) {
                        /** Drawing only objects when >confidence_threshold probability **/
                        std::ostringstream conf;
                        conf << ":" << std::fixed << std::setprecision(3) << confidence;
                        cv::putText(frame,
                                 (label < static_cast<int>(labels.size()) ?
                                        labels[label] : std::string("label #") + std::to_string(label)) + conf.str(),
                                    cv::Point2f(object.xmin, object.ymin - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                        cv::rectangle(frame, cv::Point2f(object.xmin-(((object.xmax-object.xmin)/0.8)*0.22), object.ymin-(((object.ymax-object.ymin)/0.8)*0.22)), cv::Point2f(object.xmax+(((object.xmax-object.xmin)/0.8)*0.22), object.ymax+(((object.ymax-object.ymin)/0.8)*0.22)), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                    }
                }
            }
           // cv::imshow("Detection results", frame);

  // ---------------------------lane_detect--------------------------------------------------




           Mat src=frame;
           Mat gray ;
           cvtColor(src,gray,COLOR_BGR2GRAY);

           Mat grad_x, grad_y, grad;//sobel变换模型
	   Mat hls;    //HLS图形
	   vector<Mat> channels;//HLS的通道
	   Mat hchannel, lchannel, schannel;

	  Mat result;//用来保存sobel边缘检测及高斯变换后的结果
          Sobel(gray, grad_x, 0);
	  Sobel(gray, grad_y, 1);
	  bitwise_and(grad_x, grad_y, grad);

	  cv::cvtColor(src, hls, cv::COLOR_RGB2HLS);
          split(hls, channels);
	  hchannel = channels.at(0);
	  lchannel = channels.at(1);
	  schannel = channels.at(2);
	  Mat hlsbianry = Mat::zeros(hls.rows, hls.cols, CV_8UC1);
       int i, j,m,n;

	int sthresh_1=140;
	int lthresh_1=120;
	for (i = 0; i < src.rows; i++)     //列
		for (j = 0; j < src.cols; j++)  //行
		{
			if (sthresh_1 < schannel.at<uchar>(i, j) &&lthresh_1 < lchannel.at<uchar>(i, j) )
			{
				hlsbianry.at<uchar>(i, j) = 255;
			}
			else
			{
				hlsbianry.at<uchar>(i, j) = 0;
			}
		}

	addWeighted(hlsbianry, 1, grad,1, 0.0, result);//得到的结果并不是二值图
        Mat dst_warp= Mat::zeros(hls.rows, hls.cols, CV_8UC1) ;
        warp_image(result, dst_warp,0);
        Mat   paintX;
	paintX = Mat::ones(dst_warp.rows, dst_warp.cols, CV_8UC1);
	int* v = new int[dst_warp.cols * 4];//*char转int

	memset(v, 0, dst_warp.cols * 4);
	///**************************************************************
	for (i = 0; i < dst_warp.cols; i++)           //列  
	{
		for (j = 0; j < dst_warp.rows; j++)                //行  
		{
			if (dst_warp.at<uchar>(j, i) == 0)                //统计的是黑色像素的数量  
				v[i]++;   //v[i]存到就是对应列的元素值
		}
	}
	int partition_x_pre=0, partition_x_aft=0;

	int x1 = (*min_element(v, v + paintX.cols / 2));
	int x2 = (*min_element( v + paintX.cols / 2, v + paintX.cols));
        for (i = 0; i < paintX.cols/2; i++)
	{

			if (v[i] ==x1)
			{
				partition_x_pre = i;//前半窗的车道线x位置
			//	cout << "partition_x_pre have changed to " << partition_x_pre << endl;
				circle(paintX, Point2f(i, 50) , 10, Scalar(0,0,0)); 

			}
	}

	for (j = paintX.cols / 2; j < paintX.cols; j++)
	{
		if (v[j] ==x2)
		{
			partition_x_aft = j;//后半窗的车道线x位置

		//	cout << "partition_x_aft have changed to " << partition_x_aft << endl;
			circle(paintX, Point2f(j, 50), 10, Scalar(0, 0, 0));

		}
	}

        int leftx_current = partition_x_pre;
	int rightx_current = partition_x_aft;

	int margin = 100, minpix = 500,nwindows=5;
	int num_right=0, num_left=0;

	vector<Point> points_right;//存储左拟合点
	vector<Point> points_left;//存储右拟合点

	Mat A,B;
	polynomial_curve_fit(points_right, 3, A);
	polynomial_curve_fit(points_left, 3, B);
	for(i=1;i<=nwindows;i++)
	{ 
		//cout <<i <<endl;
	rectangle(dst_warp, Point(leftx_current - margin, dst_warp.rows - i*(dst_warp.rows/ nwindows)), Point(leftx_current + margin, dst_warp.rows - (i-1) * (dst_warp.rows / nwindows)), Scalar(255, 255, 255));//黄色矩形镶边
	rectangle(dst_warp, Point(rightx_current - margin, dst_warp.rows - i * (dst_warp.rows / nwindows)), Point(rightx_current + margin, dst_warp.rows - (i - 1) * (dst_warp.rows / nwindows)), Scalar(255, 255, 255));//黄色矩形镶边
for (n = dst_warp.rows - i * (dst_warp.rows / nwindows); n < dst_warp.rows - (i - 1) * (dst_warp.rows / nwindows); n++)//对应的行坐标
	{
    	for ( m = leftx_current - margin; m< leftx_current + margin;m++)//对应的列坐标
		{
		
			if (dst_warp.at<uchar>(n, m) !=0)
			{
				num_left++;
			}
			if(num_left> minpix&& dst_warp.at<uchar>(n, m) >=200)
			{ 
			points_left.push_back(cv::Point(n ,m));								//得到二次函数拟合点,这里注意行n是变量X，列m是变量Y
			}
		}
    }
	//cout << num_left << endl;
	num_left = 0;
	}
	for (i = 1; i <= nwindows; i++)
	{
		//cout << i << endl;
		//cout << dst_warp.rows - i * (dst_warp.rows / nwindows) << endl;
		//cout << dst_warp.rows - (i-1) * (dst_warp.rows / nwindows) << endl;
		//cout << leftx_current - margin << endl;
		//cout << leftx_current + margin << endl;
		for (n = dst_warp.rows - i * (dst_warp.rows / nwindows); n < dst_warp.rows - (i - 1) * (dst_warp.rows / nwindows); n++)//对应的行坐标
		{
			for (m = rightx_current - margin; m < rightx_current + margin; m++)//对应的列坐标
			{

				if (dst_warp.at<uchar>(n, m) != 0)
				{
					num_right++;
				}
				if (num_right > minpix && dst_warp.at<uchar>(n, m) >= 200)
				{
					points_right.push_back(cv::Point(n, m));								//得到二次函数拟合点,这里注意行n是变量X，列m是变量Y
				}
			}
		}
		//cout << num_right << endl;
		num_right = 0;
	}

	Mat mask = Mat::zeros(dst_warp.rows, dst_warp.cols, CV_8UC3);
polynomial_curve_fit(points_left, 2, A);//拟合二次曲线A
	polynomial_curve_fit(points_right, 2, B);//拟合二次曲线A
	//std::cout << "A = " << A << std::endl;//拟合得到的方程
	std::vector<cv::Point> points_fitted_left;//拟合的线的点
	std::vector<cv::Point> points_fitted_right;//拟合的线的点


	for (i = 0; i < mask.rows; i++)
	{
		//先对行进行输入
		int y1 = A.at<double>(0, 0) + A.at<double>(1, 0) * i +
			A.at<double>(2, 0)*std::pow(i, 2);
		int y2 = B.at<double>(0, 0) + B.at<double>(1, 0) * i +
			B.at<double>(2, 0)*std::pow(i, 2);

		for (j = 0; j < mask.cols; j++)//对应的列坐标
		{
	
			if(y1<j&&j<y2&&result.at<uchar>(i, j)==0)//判断是否在车道线中间
			{ 
			mask.at<uchar>(i, 3*j+2)= 0;
			mask.at<uchar>(i, 3*j+1) = 255;
			mask.at<uchar>(i, 3*j+3) = 0;
			//mask.at<Vec3f>(j, i)[1] = (255, 255, 255);
			//mask.at<Vec3f>(j, i)[2] = (0, 0, 0);
			}
		}

	}
	Mat mask_return = Mat::zeros(src.rows, src.cols, CV_8UC3);
	warp_image(mask, mask_return, 1);
	Mat src_result;
	addWeighted(src, 1, mask_return, 0.3, 0.0, src_result);
            //imshow("src",src);
            imshow("src_result",src_result);

            t1 = std::chrono::high_resolution_clock::now();
            ocv_render_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            if (isLastFrame) {
                break;
            }

            if (isModeChanged) {
                isModeChanged = false;
            }


            // Final point:
            // in the truly Async mode, we swap the NEXT and CURRENT requests for the next iteration
            frame = next_frame;
            next_frame = cv::Mat();
            if (isAsyncMode) {
                async_infer_request_curr.swap(async_infer_request_next);
            }

            const int key = cv::waitKey(1);
            if (27 == key)  // Esc
                break;
            if (9 == key) {  // Tab
                isAsyncMode ^= true;
                isModeChanged = true;
            }
        }
        // -----------------------------------------------------------------------------------------------------
        auto total_t1 = std::chrono::high_resolution_clock::now();
        ms total = std::chrono::duration_cast<ms>(total_t1 - total_t0);
        //std::cout << "Total Inference time: " << total.count() << std::endl;

        /** Showing performace results **/
        if (FLAGS_pc) {
            printPerformanceCounts(*async_infer_request_curr, std::cout);
        }
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
