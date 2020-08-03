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

#include "Yolo3Detection.h"

// Namespace for using pylon objects.
using namespace Pylon;

// Namespace for using cout.
using namespace std;

CInstantCamera camera;

const enum cv::InterpolationFlags interpMode = cv::INTER_CUBIC;

tk::dnn::Yolo3Detection yolo;
tk::dnn::DetectionNN *detNN;
std::vector<tk::dnn::box> bbox;

static const int n_classes = 3;
static const int n_batch = 1;

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

    // tkdnn setup
    detNN = &yolo;
    assert(detNN->init("../models/yolo4_cones_int8.rt", n_classes, n_batch));

    try
    {

        CTlFactory& TlFactory = CTlFactory::GetInstance();
        CDeviceInfo di;
        di.SetFriendlyName("CameraLeft (40022599)");
        camera.Attach(TlFactory.CreateDevice(di));

        // Print the model name of the camera.
        cout << "Using device " << camera.GetDeviceInfo().GetModelName() << endl;

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
                cout << "Nearest Interp" << endl;
                break;
            case (cv::INTER_LINEAR):
                cout << "Linear Interp" << endl;
                break;
            case (cv::INTER_CUBIC):
                cout << "Cubic Interp" << endl;
                break;
            default:
                cout << "Invalid Interp" << endl;
                return 1;
        }

        if (USE_CUDA) {
            cout << "Using CUDA" << endl;
        }
        else {
            cout << "CPU with " << numCores << " threads" << endl;
        }

        // Start the grabbing of c_countOfImagesToGrab images.
        // The camera device is parameterized with a default configuration which
        // sets up free-running continuous acquisition.
        camera.StartGrabbing(GrabStrategy_LatestImageOnly);

        // This smart pointer will receive the grab result data.
        CGrabResultPtr ptrGrabResult;

        // Camera.StopGrabbing() is called automatically by the RetrieveResult() method
        // when c_countOfImagesToGrab images have been retrieved.

        auto now = chrono::high_resolution_clock::now();

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
                now = chrono::high_resolution_clock::now();
                auto deltaT = chrono::duration_cast<chrono::microseconds>(now - then).count();

                cout << "Image ID: " << ptrGrabResult->GetImageNumber();
                cout << "\tReal Frame Rate: " << setw(10) << 1e6/deltaT;

                int height = (int) ptrGrabResult->GetHeight();
                int width = (int) ptrGrabResult->GetWidth();

                cv::Mat inMat = cv::Mat(height, width, CV_8UC1, static_cast<uint8_t *>(ptrGrabResult->GetBuffer()));
                cv::Mat unDist = cv::Mat(height, width, CV_8UC1);
                cv::Mat Mat_RGB = cv::Mat(height, width, CV_8UC3);

                if (USE_CUDA) {
                    cv::cuda::GpuMat src, dst;
                    src.upload(inMat);
                    
                    cv::cuda::cvtColor(src, dst, cv::COLOR_BayerRG2BGR);
                    cv::cuda::remap(dst, src, map1_cuda, map2_cuda, interpMode);

                    src.download(unDist);
                }
                else {
                    cv::cvtColor(inMat, Mat_RGB, cv::COLOR_BayerRG2BGR);
                    cv::remap(Mat_RGB, unDist, map1, map2, interpMode);
                }

                auto now2 = chrono::high_resolution_clock::now();
                deltaT = chrono::duration_cast<chrono::microseconds>(now2 - now).count();

                cout << "\tFrame Time (us): "  << setw(10) << deltaT;

                std::vector<cv::Mat> batch_frame;
                std::vector<cv::Mat> batch_dnn_input;

                // batch frame will be used for image output
                batch_frame.push_back(unDist);

                // dnn input will be resized to network format
                batch_dnn_input.push_back(unDist.clone());

                // network inference
                detNN->update(batch_dnn_input, n_batch);
                detNN->draw(batch_frame);

                cout << " " << setw(5) << detNN->batchDetected[0].size() << " objects detected.";

                cv::imshow("Camera_Undist", batch_frame[0]);
                cv::resizeWindow("Camera_Undist", 1200, 600);

                cv::waitKey(1);

                cout << endl;
            }
            else
            {
                cout << "Error: " << ptrGrabResult->GetErrorCode() << " " << ptrGrabResult->GetErrorDescription() << endl;
            }
        }
    }
    catch (const GenericException &e)
    {
        // Error handling.
        cerr << "An exception occurred." << endl
        << e.GetDescription() << endl;
        exitCode = 1;
    }

    // // Comment the following two lines to disable waiting on exit.
    // cerr << endl << "Press enter to exit." << endl;
    // while( cin.get() != '\n');

    Pylon::PylonTerminate();

    return exitCode;
}