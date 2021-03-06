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

// Namespace for using pylon objects.
using namespace Pylon;

// Namespace for using cout.
using namespace std;

CInstantCamera camera;

const enum cv::InterpolationFlags interpMode = cv::INTER_CUBIC;

void *ExposureLoop(void *) {
    // GenApi::INodeMap& nodemap = camera.GetNodeMap();
    // while (true) {
    //     for (int i = 10; i < 20; i += 2) {
    //         CFloatParameter(nodemap, "ExposureTime").SetValue(i*1000);

    //         sleep(1);
    //     }
    // }
}

int main(int argc, char** argv) {
    int exitCode;

    const bool USE_CUDA = true;
    int numCores = -1;

    if (argc > 1) {
        numCores = atoi(argv[1]);
    }

    cv::setNumThreads(numCores);
    Pylon::PylonInitialize();

    cv::FileStorage fs;
    fs.open("../calibration.xml", cv::FileStorage::READ);

    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;

    fs["cameraMatrix"] >> cameraMatrix;
    fs["distCoeffs"] >> distCoeffs;

    fs.release();

    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, cv::Size(1920, 1200), 0);
    cv::Mat map1, map2;
    cv::cuda::GpuMat map1_cuda, map2_cuda;

    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), newCameraMatrix, cv::Size(1920, 1200), CV_32FC1, map1, map2);
    map1_cuda.upload(map1);
    map2_cuda.upload(map2);

    std::cout << "Loaded calibration file" << std::endl;

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
        CEnumParameter(nodemap, "PixelFormat").SetValue("BayerBG10");
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
        camera.StartGrabbing(100, GrabStrategy_LatestImageOnly);

        // This smart pointer will receive the grab result data.
        CGrabResultPtr ptrGrabResult;

        pthread_t thread;
        pthread_create(&thread, NULL, ExposureLoop, NULL);

        // Camera.StopGrabbing() is called automatically by the RetrieveResult() method
        // when c_countOfImagesToGrab images have been retrieved.

        auto now = chrono::high_resolution_clock::now();

        cv::namedWindow("Camera_Undist", 0);
        cv::namedWindow("Camera_Raw", 0);

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
                cout << "\tReal Frame Time (us): " << setw(10) << deltaT;

                int height = (int) ptrGrabResult->GetHeight();
                int width = (int) ptrGrabResult->GetWidth();

                cv::Mat inMat = cv::Mat(height, width, CV_16UC1, (uint16_t *) ptrGrabResult->GetBuffer());
                cv::Mat unDist = cv::Mat(height, width, CV_16UC1);
                cv::Mat Mat16_RGB = cv::Mat(height, width, CV_16UC3);

                if (USE_CUDA) {
                    cv::cuda::GpuMat src, dst;
                    src.upload(inMat);
                    
                    cv::cuda::multiply(src, 64, dst);
                    cv::cuda::cvtColor(dst, src, cv::COLOR_BayerRG2BGR);
                    cv::cuda::remap(src, dst, map1_cuda, map2_cuda, interpMode);

                    // src.download(Mat16_RGB);
                    dst.download(unDist);
                }
                else {
                    inMat = inMat.mul(64);

                    cv::cvtColor(inMat, Mat16_RGB, cv::COLOR_BayerRG2BGR);
                    cv::remap(Mat16_RGB, unDist, map1, map2, interpMode);
                }

                auto now2 = chrono::high_resolution_clock::now();
                deltaT = chrono::duration_cast<chrono::microseconds>(now2 - now).count();

                cout << "\tFrame Time (us): "  << setw(10) << deltaT << endl;

                // cv::imshow("Camera_Undist", unDist);
                // cv::resizeWindow("Camera_Undist", 1200, 600);

                // cv::imshow("Camera_Raw", Mat16_RGB);
                // cv::resizeWindow("Camera_Raw", 1200, 600);

                // cv::waitKey(1);

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