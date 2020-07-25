#include <pylon/PylonIncludes.h>
#include <GenApi/GenApi.h>
#include <iostream>
#include <pthread.h>
#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
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

const bool USE_CUDA = true;

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

    Pylon::PylonAutoInitTerm autoInitTerm;

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

        GenApi::INodeMap& nodemap = camera.GetNodeMap();

        cout << "test" << endl;

        // Start the grabbing of c_countOfImagesToGrab images.
        // The camera device is parameterized with a default configuration which
        // sets up free-running continuous acquisition.
        camera.StartGrabbing(GrabStrategy_LatestImageOnly);

        // This smart pointer will receive the grab result data.
        CGrabResultPtr ptrGrabResult;

        pthread_t thread;
        pthread_create(&thread, NULL, ExposureLoop, NULL);

        // Camera.StopGrabbing() is called automatically by the RetrieveResult() method
        // when c_countOfImagesToGrab images have been retrieved.

        auto now = chrono::high_resolution_clock::now();

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
                auto deltaT = chrono::duration_cast<chrono::milliseconds>(now - then).count();

                cout << "Image ID: " << ptrGrabResult->GetImageNumber();
                cout << ", FPS: " << 1000.0/deltaT << endl;

                cv::Mat inMat = cv::Mat((int) ptrGrabResult->GetHeight(), (int) ptrGrabResult->GetWidth(), CV_16UC1, (uint16_t *) ptrGrabResult->GetBuffer());
                cv::Mat Mat16_RGB = cv::Mat((int) ptrGrabResult->GetHeight(), (int) ptrGrabResult->GetWidth(), CV_16UC3);

                if (USE_CUDA) {
                    cv::cuda::GpuMat src, dst;
                    src.upload(inMat);
                    cv::cuda::multiply(src, 64, dst);
                    cv::cuda::cvtColor(dst, src, cv::COLOR_BayerRG2BGR);
                    src.download(Mat16_RGB);
                }
                else {
                    inMat = inMat.mul(16);
                    cv::cvtColor(inMat, Mat16_RGB, cv::COLOR_BayerRG2BGR);
                }


                cv::imshow("image", Mat16_RGB);
                cv::waitKey(1);
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

    // Comment the following two lines to disable waiting on exit.
    cerr << endl << "Press enter to exit." << endl;
    while( cin.get() != '\n');

    return exitCode;
}