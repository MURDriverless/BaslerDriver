#include <pylon/PylonIncludes.h>
#include <GenApi/GenApi.h>
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

#include "Detectors.hpp"

// Namespace for using pylon objects.
using namespace Pylon;

CInstantCamera camera;

const enum cv::InterpolationFlags interpMode = cv::INTER_CUBIC;

int main(int argc, char** argv) {
    int exitCode;

    const bool USE_CUDA = true;
    int numCores = -1;

    if (argc > 1) {
        numCores = atoi(argv[1]);
    }

    cv::setNumThreads(numCores);

    // Camera Setup
    Pylon::PylonInitialize();

    cv::FileStorage fs;
    fs.open("../calibration.xml", cv::FileStorage::READ);

    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;

    fs["cameraMatrix"] >> cameraMatrix;
    fs["distCoeffs"] >> distCoeffs;

    fs.release();

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

    try
    {

        CTlFactory& TlFactory = CTlFactory::GetInstance();
        CDeviceInfo di;
        di.SetFriendlyName("CameraLeft (40022599)");
        camera.Attach(TlFactory.CreateDevice(di));

        // Print the model name of the camera.
        std::cout << "Using device " << camera.GetDeviceInfo().GetModelName() << std::endl;

        // The parameter MaxNumBuffer can be used to control the count of buffers
        // allocated for grabbing. The default value of this parameter is 10.
        // camera.MaxNumBuffer = 1;

        camera.Open();
        GenApi::INodeMap& nodemap = camera.GetNodeMap();
        CEnumParameter(nodemap, "PixelFormat").SetValue("BayerBG8");
        CBooleanParameter(nodemap, "AcquisitionFrameRateEnable").SetValue(false);
        camera.Close();

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

        // Start the grabbing of c_countOfImagesToGrab images.
        // The camera device is parameterized with a default configuration which
        // sets up free-running continuous acquisition.
        camera.StartGrabbing(GrabStrategy_LatestImageOnly);

        // This smart pointer will receive the grab result data.
        CGrabResultPtr ptrGrabResult;

        // Camera.StopGrabbing() is called automatically by the RetrieveResult() method
        // when c_countOfImagesToGrab images have been retrieved.

        auto now = std::chrono::high_resolution_clock::now();

        cv::namedWindow("Camera_Undist", 0);

        while ( camera.IsGrabbing())
        {
            // Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            camera.RetrieveResult( 5000, ptrGrabResult, TimeoutHandling_ThrowException);

            // Image grabbed successfully?
            if (ptrGrabResult->GrabSucceeded())
            {
                // Access the image data.
                auto then = now;
                now = std::chrono::high_resolution_clock::now();
                auto deltaT = std::chrono::duration_cast<std::chrono::microseconds>(now - then).count();

                std::cout << "Image ID: " << ptrGrabResult->GetImageNumber();
                std::cout << "\tReal Frame Rate: " << std::setw(10) << 1e6/deltaT;

                int height = (int) ptrGrabResult->GetHeight();
                int width = (int) ptrGrabResult->GetWidth();

                cv::Mat inMat = cv::Mat(height, width, CV_8UC1, static_cast<uint8_t *>(ptrGrabResult->GetBuffer()));
                cv::Mat unDist = cv::Mat(height, width, CV_8UC1);
                cv::Mat Mat_RGB = cv::Mat(height, width, CV_8UC3);

                cv::cuda::GpuMat src, dst;
                src.upload(inMat);
                
                cv::cuda::cvtColor(src, dst, cv::COLOR_BayerRG2BGR);
                cv::cuda::remap(dst, src, map1_cuda, map2_cuda, interpMode);

                // src.download(unDist);

                auto now2 = std::chrono::high_resolution_clock::now();
                deltaT = std::chrono::duration_cast<std::chrono::microseconds>(now2 - now).count();

                std::cout << "\tFrame Time (us): "  << std::setw(10) << deltaT;

                detectors.detectFrame(src);
            }
            else
            {
                std::cout << "Error: " << ptrGrabResult->GetErrorCode() << " " << ptrGrabResult->GetErrorDescription() << std::endl;
            }
        }
    }
    catch (const GenericException &e)
    {
        // Error handling.
        std::cerr << "An exception occurred." << std::endl
        << e.GetDescription() << std::endl;
        exitCode = 1;
    }

    Pylon::PylonTerminate();


    return exitCode;
}