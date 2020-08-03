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
#include "GeniPylon.hpp"

// Namespace for using cout.
using namespace std;

const enum cv::InterpolationFlags interpMode = cv::INTER_CUBIC;

int main(int argc, char** argv) {
    int exitCode;

    const bool USE_CUDA = true;
    int numCores = -1;

    if (argc > 1) {
        numCores = atoi(argv[1]);
    }

    cv::setNumThreads(numCores);
    
    IGeniCam* camera = new PylonCam();
    camera->initializeLibrary();

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
        camera->setup("CameraLeft (40022599)");

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

        camera->startGrabbing();

        auto now = chrono::high_resolution_clock::now();

        cv::namedWindow("Camera_Undist", 0);
        cv::namedWindow("Camera_Raw", 0);

        int imageCount = 0;

        while ( camera->isGrabbing())
        {
            int height, width;
            uint8_t* buffer;

            bool ret = camera->retreiveResult(height, width, buffer);

            // Image grabbed successfully?
            if (ret)
            {
                // Access the image data.
                auto then = now;
                now = chrono::high_resolution_clock::now();
                auto deltaT = chrono::duration_cast<chrono::microseconds>(now - then).count();

                cout << "Image ID: " << imageCount++;
                cout << "\tReal Frame Time (us): " << setw(10) << deltaT;

                cv::Mat inMat = cv::Mat(height, width, CV_8UC1, buffer);
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

                cout << "\tFrame Time (us): "  << setw(10) << deltaT << endl;

                cv::imshow("Camera_Undist", unDist);
                cv::resizeWindow("Camera_Undist", 1200, 600);

                // cv::imshow("Camera_Raw", Mat16_RGB);
                // cv::resizeWindow("Camera_Raw", 1200, 600);

                cv::waitKey(1);

                camera->clearResult();
            }
            else
            {
                cout << "Error" << endl;
            }
        }
    }
    catch (const exception &e)
    {
        // Error handling.
        cerr << "An exception occurred." << endl << e.what() << endl;
        exitCode = 1;
    }

    camera->finalizeLibrary();
    free(camera);

    return exitCode;
}