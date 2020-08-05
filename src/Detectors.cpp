#include "Detectors.hpp"

Detectors::Detectors() {

}

Detectors::~Detectors() {
    delete detNN.release();
    delete keypointDetector.release();
}

void Detectors::initialize(std::string objectModel, std::string featureModel) {

    detNN.reset(new tk::dnn::Yolo3Detection());
    detNN->init(objectModel, n_classes, n_batch);

    std::string trtPath = featureModel.replace(featureModel.end() - 4,
                                               featureModel.end(), "trt");

    keypointDetector.reset(
    new KeypointDetector(
        featureModel,
        trtPath, 
        keypointsW, 
        keypointsH, 
        maxBatch)
    );

    detNN.reset(new tk::dnn::Yolo3Detection());
    detNN->init(objectModel, n_classes, n_batch);
}

void Detectors::detectFrame(const cv::cuda::GpuMat imageFrameGpu) {
    cv::Mat imageFrame;
    imageFrameGpu.download(imageFrame);

    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;

    // batch frame will be used for image output
    batch_frame.push_back(imageFrame);

    // dnn input will be resized to network format
    batch_dnn_input.push_back(imageFrame.clone());

    // network inference
    detNN->update(batch_dnn_input, n_batch);

    std::vector<tk::dnn::box> bboxs = detNN->batchDetected[0];

    // generate a vector of image crops for keypoint detector
    std::vector<cv::Mat> rois;

    for (const auto &bbox: bboxs)
    {
        int left    = std::max(double(bbox.x), 0.0);
        int right   = std::min(double(bbox.x + bbox.w), (double) imageFrame.cols);
        int top     = std::max(double(bbox.y), 0.0);
        int bot     = std::min(double(bbox.y + bbox.h), (double) imageFrame.rows);

        cv::Rect box(cv::Point(left, top), cv::Point(right, bot));
        cv::Mat roi = imageFrame(box);
        rois.push_back(roi);
    }

    // keypoint network inference
    std::vector<std::vector<cv::Point2f>> keypoints = keypointDetector->doInference(rois);

    detNN->draw(batch_frame);
    for (int i = 0; i < bboxs.size(); i++) {
        for (auto &keypoint : keypoints[i]) {
            keypoint.y += bboxs[i].y;
            keypoint.x += bboxs[i].x;

            cv::circle(batch_frame[0], keypoint, 3, cv::Scalar(0, 255, 0), -1, 8);
        }
    }

    cv::imshow("Camera_Undist", batch_frame[0]);
    cv::resizeWindow("Camera_Undist", 1200, 600);

    cv::waitKey(1);

    std::cout << " " << std::setw(5) << detNN->batchDetected[0].size() << " objects detected.";
    std::cout << std::endl;
}