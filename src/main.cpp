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

    fs["cameraMatrix"] >> cameraMatrix;
    fs["distCoeffs"] >> distCoeffs;

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
    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, cv::Size(1920, 1200), 0);
    cv::Mat map1, map2;
    cv::cuda::GpuMat map1_cuda, map2_cuda;

    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), newCameraMatrix, cv::Size(1920, 1200), CV_32FC1, map1, map2);
    map1_cuda.upload(map1);
    map2_cuda.upload(map2);

    std::cout << "Loaded calibration file" << std::endl;

    Detectors detectors;
    detectors.initialize("../models/yolo4_cones_int8.rt", "../models/keypoints.onnx");

    std::unique_ptr<IGeniCam> camera;
    camera.reset(IGeniCam::create(GeniImpl::Pylon_i));
    camera->initializeLibrary();

    cv::cuda::setBufferPoolUsage(true);
    cv::cuda::setBufferPoolConfig(cv::cuda::getDevice(), 1920 * 1200 * 8, 2);

    cv::cuda::Stream cam1Stream;
    cv::cuda::Stream cam2Stream;
    cv::cuda::BufferPool pool1(cam1Stream), pool2(cam2Stream);

    try
    {
        camera->setup("CameraLeft (40022599)");

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

        cv::namedWindow("Camera_Undist", 0);
        cv::namedWindow("Camera_Undist2", 0);

        unsigned int grabCount = 6000;
        unsigned int imageID = 0;

        camera->startGrabbing(grabCount);

        while ((imageID < grabCount) && camera->isGrabbing())
        {
            // Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            int height;
            int width;
            uint8_t* buffer;
            bool ret = camera->retreiveResult(height, width, buffer);

            // Image grabbed successfully?
            if (ret)
            {
                // Access the image data.
                auto then = now;
                now = std::chrono::high_resolution_clock::now();
                auto deltaT = std::chrono::duration_cast<std::chrono::microseconds>(now - then).count();

                std::cout << "Image ID: " << imageID++;
                std::cout << "\tReal Frame Rate: " << std::setw(10) << 1e6/deltaT;

                cv::Mat inMat_t = cv::Mat(height, width, CV_8UC1, buffer);
                // cv::Mat unDist = cv::Mat(height, width, CV_8UC1);

                cv::cuda::HostMem inMat(height, width, CV_8UC1, cv::cuda::HostMem::PAGE_LOCKED);
                inMat_t.copyTo(inMat);

                cv::cuda::HostMem unDist1(height, width, CV_8UC1);
                cv::cuda::HostMem unDist2(height, width, CV_8UC1);

                cv::cuda::GpuMat src1 = pool1.getBuffer(1920, 1200, CV_8UC1);
                cv::cuda::GpuMat rgb1 = pool1.getBuffer(1920, 1200, CV_8UC3);
                cv::cuda::GpuMat uDist1 = pool1.getBuffer(1920, 1200, CV_8UC3);

                cv::cuda::GpuMat src2 = pool1.getBuffer(1920, 1200, CV_8UC1);
                cv::cuda::GpuMat rgb2 = pool1.getBuffer(1920, 1200, CV_8UC3);
                cv::cuda::GpuMat uDist2 = pool1.getBuffer(1920, 1200, CV_8UC3);


                src1.upload(inMat, cam1Stream);
                src2.upload(inMat, cam2Stream);

                cv::cuda::cvtColor(src1, rgb1, cv::COLOR_BayerRG2BGR, 0, cam1Stream);
                cv::cuda::remap(rgb1, uDist1, map1_cuda, map2_cuda, interpMode, 0, cv::Scalar(), cam1Stream);

                // cam1Stream.waitForCompletion();

                cv::cuda::cvtColor(src2, rgb2, cv::COLOR_BayerRG2BGR, 0, cam2Stream);
                cv::cuda::remap(rgb2, uDist2, map1_cuda, map2_cuda, interpMode, 0, cv::Scalar(), cam2Stream);

                auto now2 = std::chrono::high_resolution_clock::now();
                deltaT = std::chrono::duration_cast<std::chrono::microseconds>(now2 - now).count();

                std::cout << "\tFrame Time (us): "  << std::setw(10) << deltaT << " ";

                std::vector<ConeROI> coneROIs;
                detectors.detectFrame(uDist1, coneROIs);

                uDist2.download(unDist2, cam2Stream);
                cam2Stream.waitForCompletion();

                for (int i = 0; i < coneROIs.size(); i++) {
                    if (i > 1) {
                        break;
                    }

                    ConeROI &coneROI = coneROIs[i];
                    cv::Mat tvec;
                    cv::Mat rvec;

                    std::vector<cv::Point3f> cone4 (conePoints.begin()+1, conePoints.end()-2);
                    std::vector<cv::Point2f> key4  (coneROI.keypoints.begin()+1, coneROI.keypoints.end()-2);

                    bool ret = cv::solvePnP(cone4, key4, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SolvePnPMethod::SOLVEPNP_IPPE);
                    if (true) {
                        std::stringstream outputString;
                        outputString << std::setfill('0') << std::setw(8) 
                        << std::fixed << std::setprecision(2) <<
                        tvec.at<double>(2,0) << std::endl;
                        cv::putText(unDist2, outputString.str(), cv::Point(1000, 600), cv::HersheyFonts::FONT_HERSHEY_PLAIN, 5, CV_RGB(118, 185, 0), 2);

                        std::cout << "Est Depth: " << tvec.at<double>(2, 0) << std::endl;
                    }
                }

                cv::imshow("Camera_Undist2", unDist2);
                cv::resizeWindow("Camera_Undist2", 1200, 600);

                cv::waitKey(1);

                camera->clearResult();
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

    camera->finalizeLibrary();
    delete camera.release();

    return 0;
}