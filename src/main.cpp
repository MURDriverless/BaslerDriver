#include <iostream>
#include <pthread.h>
#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <chrono>

#include "GeniWrap.hpp"

#include "Detectors.hpp"

const enum cv::InterpolationFlags interpMode = cv::INTER_LINEAR;

int main(int argc, char** argv) {
    int exitCode;

    const bool USE_CUDA = true;
    int numCores = -1;

    if (argc > 1) {
        numCores = atoi(argv[1]);
    }

    cv::setNumThreads(numCores);

    cv::FileStorage fs;
    fs.open("../calibration.xml", cv::FileStorage::READ);

    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    cv::Size calibSize;

    fs["cameraMatrix"] >> cameraMatrix;
    fs["distCoeffs"] >> distCoeffs;
    fs["calibImageSize"] >> calibSize;

    fs.release();

    std::vector<cv::Point3f> conePoints; // Real world mm
    conePoints.push_back(cv::Point3f(0, 0, 0));

    for (int i = 1; i <= 3; i++) {
        float x = -77.5/3.0f * i;
        float y = 300.0f/3.0f * i;

        conePoints.push_back(cv::Point3f( x, y, 0));
        conePoints.push_back(cv::Point3f(-x, y, 0));
    }

    // OpenCV Setup
    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, calibSize, 0, cv::Size());
    cv::Mat map1, map2;
    cv::cuda::GpuMat map1_cuda, map2_cuda;

    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), newCameraMatrix, cv::Size(), CV_32FC1, map1, map2);
    map1_cuda.upload(map1);
    map2_cuda.upload(map2);

    std::cout << "Loaded calibration file" << std::endl;

    cv::Ptr<cv::Feature2D> featureDetector = cv::xfeatures2d::SIFT::create();
    cv::Ptr<cv::DescriptorMatcher> descriptorMatcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

    // Detector Setup
    Detectors detectors;
    detectors.initialize("../models/yolo4_cones_int8.rt", "../models/keypoints.onnx");

    // Camera Setup
    std::unique_ptr<IGeniCam> camera1;
    std::unique_ptr<IGeniCam> camera2;

    camera1.reset(IGeniCam::create(GeniImpl::Pylon_i));
    camera2.reset(IGeniCam::create(GeniImpl::Pylon_i));
    
    camera1->initializeLibrary();

    cv::cuda::setBufferPoolUsage(true);
    cv::cuda::setBufferPoolConfig(cv::cuda::getDevice(), 1920 * 1200 * 8, 2);

    cv::cuda::Stream cam1Stream;
    cv::cuda::Stream cam2Stream;
    cv::cuda::BufferPool pool1(cam1Stream), pool2(cam2Stream);

    try
    {
        camera1->setup("CameraLeft (40022599)");
        camera2->setup("CameraRight (22954692)");
        // camera1->setup("CameraLeft (40022599)");

        switch (interpMode) {
            case (cv::INTER_NEAREST):
                std::cout << "Nearest Interp" << std::endl;
                break;
            case (cv::INTER_LINEAR):
                std::cout << "Linear Interp" << std::endl;
                break;
            case (cv::INTER_CUBIC):
                std::cout << "Cubic Interp" << std::endl;
                break;
            default:
                std::cout << "Invalid Interp" << std::endl;
                return 1;
        }

        if (USE_CUDA) {
            std::cout << "Using CUDA" << std::endl;
            std::cout << "OPENCV_CUDACONTRIB Flag: ";

            #ifdef OPENCV_CUDACONTRIB
            std::cout << "True" << std::endl;
            #else
            std::cout << "False" << std::endl;
            #endif
        }
        else {
            std::cout << "Requires to be compiled with CUDA" << std::endl;
            return 1;
        }

        auto now = std::chrono::high_resolution_clock::now();

        cv::namedWindow("Camera_Undist1", 0);
        cv::namedWindow("Camera_Undist2", 0);

        cv::namedWindow("Cam1_crop", 0);
        cv::namedWindow("Cam2_crop", 0);

        cv::namedWindow("Matches", 0);

        unsigned int grabCount = 6000;
        unsigned int imageID = 0;

        camera1->startGrabbing(grabCount);
        camera2->startGrabbing(grabCount);

        while ((imageID < grabCount) && camera1->isGrabbing() && camera2->isGrabbing())
        {
            // Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            int height1;
            int width1;
            uint8_t* buffer1;

            int height2;
            int width2;
            uint8_t* buffer2;

            bool ret1 = camera1->retreiveResult(height1, width1, buffer1);
            bool ret2 = camera2->retreiveResult(height2, width2, buffer2);

            // Image grabbed successfully?
            if (ret1 && ret2)
            {
                // Access the image data.
                auto then = now;
                now = std::chrono::high_resolution_clock::now();
                auto deltaT = std::chrono::duration_cast<std::chrono::microseconds>(now - then).count();

                std::cout << "Image ID: " << imageID++;
                std::cout << "\tReal Frame Rate: " << std::setw(10) << 1e6/deltaT;

                cv::Mat inMat = cv::Mat(height1, width1, CV_8UC1, buffer1);
                cv::Mat inMat2 = cv::Mat(height2, width2, CV_8UC1, buffer2);

                cv::Mat unDist1;
                cv::Mat unDist2;

                cv::cuda::GpuMat src1   ;//= pool1.getBuffer(1920, 1200, CV_8UC1);
                cv::cuda::GpuMat rgb1   ;//= pool1.getBuffer(1920, 1200, CV_8UC3);
                cv::cuda::GpuMat uDist1 ;//= pool1.getBuffer(1920, 1200, CV_8UC3);

                cv::cuda::GpuMat src2   ;//= pool1.getBuffer(1920, 1200, CV_8UC1);
                cv::cuda::GpuMat rgb2   ;//= pool1.getBuffer(1920, 1200, CV_8UC3);
                cv::cuda::GpuMat uDist2 ;//= pool1.getBuffer(1920, 1200, CV_8UC3);


                src1.upload(inMat, cam1Stream);
                src2.upload(inMat2, cam2Stream);

                cv::cuda::cvtColor(src1, rgb1, cv::COLOR_BayerRG2BGR, 0, cam1Stream);
                // cv::cuda::remap(rgb1, uDist1, map1_cuda, map2_cuda, interpMode, 0, cv::Scalar(), cam1Stream);

                // cam1Stream.waitForCompletion();

                // cv::cuda::cvtColor(src2, rgb2, cv::COLOR_BayerRG2BGR, 0, cam2Stream);
                rgb2 = src2.clone();
                // cv::cuda::remap(rgb2, uDist2, map1_cuda, map2_cuda, interpMode, 0, cv::Scalar(), cam2Stream);

                auto now2 = std::chrono::high_resolution_clock::now();
                deltaT = std::chrono::duration_cast<std::chrono::microseconds>(now2 - now).count();

                std::cout << "\tFrame Time (us): "  << std::setw(10) << deltaT << " ";

                std::vector<ConeROI> coneROIs;

                cv::Mat temp1, temp2;

                rgb1.download(temp1, cam1Stream);
                rgb2.download(temp2, cam2Stream);

                cv::resize(temp1, unDist1, cv::Size(), 1, 1, cv::INTER_NEAREST);
                cv::resize(temp2, unDist2, cv::Size(), 1.67/4.8, 1.67/4.8, cv::INTER_NEAREST);

                cv::Mat unDist1_r = unDist1.clone();
                cv::Mat unDist2_r = unDist2.clone();

                cv::Mat unDist1_gray(1920, 1200, CV_8UC1);
                cv::Mat unDist2_gray(1920, 1200, CV_8UC1);

                cv::cvtColor(unDist1_r, unDist1_gray, cv::COLOR_BGR2GRAY);
                // cv::cvtColor(unDist2_r, unDist2_gray, cv::COLOR_BGR2GRAY);
                unDist2_gray = unDist2_r.clone();

                width2 = unDist2.cols;
                height2 = unDist2.rows;

                detectors.detectFrame(unDist1, coneROIs);

                cam2Stream.waitForCompletion();

                for (int i = 0; i < coneROIs.size(); i++) {
                    if (i > 1) {
                        break;
                    }

                    ConeROI &coneROI = coneROIs[i];
                    cv::Mat tvec;
                    cv::Mat rvec;

                    double est_depth;

                    std::vector<cv::Point3f> cone4 (conePoints.begin()+1, conePoints.end()-2);
                    std::vector<cv::Point2f> key4  (coneROI.keypoints.begin()+1, coneROI.keypoints.end()-2);

                    bool ret = cv::solvePnP(cone4, key4, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SolvePnPMethod::SOLVEPNP_IPPE);
                    if (true) {
                        std::stringstream outputString;
                        outputString << std::setfill('0') << std::setw(8) 
                        << std::fixed << std::setprecision(2) <<
                        tvec.at<double>(2,0) << std::endl;
                        cv::putText(unDist2, outputString.str(), cv::Point(1000, 600), cv::HersheyFonts::FONT_HERSHEY_PLAIN, 5, CV_RGB(118, 185, 0), 2);

                        est_depth = tvec.at<double>(2, 0);

                        const double f1 = 1041.67;
                        const double f2 = 1250;
                        const double B = 250;

                        std::cout << "Est Depth: " << est_depth << std::endl;

                        float border = 0.15f;
                        coneROI.roiRect -= cv::Point2i(border * coneROI.roiRect.width, border * coneROI.roiRect.height);
                        coneROI.roiRect += cv::Size2i(2*border * coneROI.roiRect.width, 2*border * coneROI.roiRect.height);

                        cv::Rect projRect(coneROI.roiRect);

                        int x_p = coneROI.roiRect.x - width1/2;
                        int x_pp = (f2/est_depth) * (est_depth/f1 * x_p - B);

                        projRect.x = x_pp + width2/2;

                        // projRect.x -= coneROI.roiRect.width * border;
                        // projRect.y -= coneROI.roiRect.height * border;
                        // projRect.width *= 1.0f + 2*border;
                        // projRect.height *= 1.0f + 2*border;

                        cv::rectangle(unDist2, projRect, cv::Scalar(255, 255, 255));

                        std::vector<cv::Mat> imgPair;
                        imgPair.push_back(unDist1_gray);
                        imgPair.push_back(unDist2_gray);

                        std::vector<cv::Rect> roiMask;
                        roiMask.push_back(coneROI.roiRect);
                        roiMask.push_back(projRect);

                        try {
                            cv::Mat unDist1_cropped = unDist1_gray(coneROI.roiRect);
                            // cv::Mat unDist2_cropped = unDist1_gray(coneROI.roiRect);
                            // cv::rotate(unDist1_gray(coneROI.roiRect), unDist2_cropped, cv::ROTATE_90_CLOCKWISE);
                            cv::Mat unDist2_cropped = unDist2_gray(projRect);

                            cv::imshow("Cam1_crop", unDist1_cropped);
                            cv::resizeWindow("Cam1_crop", 1000, 600);

                            cv::imshow("Cam2_crop", unDist2_cropped);
                            cv::resizeWindow("Cam2_crop", 1000, 600);

                            cv::waitKey(1);

                            std::vector<cv::KeyPoint> featureKeypoints1;
                            std::vector<cv::KeyPoint> featureKeypoints2;
                            cv::Mat descriptors1;
                            cv::Mat descriptors2;

                            std::vector<cv::DMatch> matches;
                            std::vector<cv::DMatch> matchesFilt;

                            featureDetector->detectAndCompute(unDist1_cropped, cv::noArray(), featureKeypoints1, descriptors1);
                            featureDetector->detectAndCompute(unDist2_cropped, cv::noArray(), featureKeypoints2, descriptors2);

                            descriptorMatcher->match(descriptors1, descriptors2, matches);

                            uint32_t yDelta = projRect.height * 0.1;

                            for (const cv::DMatch &match: matches) {
                                std::cout << match.queryIdx << ", " << match.trainIdx << std::endl;
                                
                                if (abs(featureKeypoints1[match.queryIdx].pt.y - featureKeypoints2[match.trainIdx].pt.y) < yDelta) {
                                    matchesFilt.push_back(match);
                                }
                            }

                            cv::Mat imgMatch;
                            cv::drawMatches(unDist1_cropped, featureKeypoints1, unDist2_cropped, featureKeypoints2, matchesFilt, imgMatch);
                            // cv::drawKeypoints(unDist1_cropped, featureKeypoints1, imgMatch);

                            cv::imshow("Matches", imgMatch);
                            cv::resizeWindow("Matches", 1000, 600);

                            cv::waitKey(1);
                        }
                        catch (const std::exception &e) {
                            // Error handling.
                            std::cerr << "An exception occurred." << std::endl
                            << e.what() << std::endl;
                        }
                    }
                }

                cv::imshow("Camera_Undist2", unDist2);
                cv::resizeWindow("Camera_Undist2", 1000, 600);

                cv::waitKey(1);

                camera1->clearResult();
                camera2->clearResult();
            }
            else
            {
                std::cout << "Failed to grab image." << std::endl;
            }
        }
    }
    catch (const std::exception &e)
    {
        // Error handling.
        std::cerr << "An exception occurred." << std::endl
        << e.what() << std::endl;
        exitCode = 1;
    }

    camera1->finalizeLibrary();
    delete camera1.release();
    delete camera2.release();

    return 0;
}