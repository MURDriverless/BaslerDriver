#include <pylon/PylonIncludes.h>
#include <GenApi/GenApi.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <pthread.h>
#include <unistd.h>

// Namespace for using pylon objects.
using namespace Pylon;

// Namespace for using cout.
using namespace std;

CInstantCamera camera;

void *ExposureLoop(void *) {
    GenApi::INodeMap& nodemap = camera.GetNodeMap();
    while (true) {
        for (int i = 10; i < 20; i += 2) {
            CFloatParameter(nodemap, "ExposureTime").SetValue(i*1000);

            sleep(1);
        }
    }
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
        camera.MaxNumBuffer = 5;

        // Start the grabbing of c_countOfImagesToGrab images.
        // The camera device is parameterized with a default configuration which
        // sets up free-running continuous acquisition.
        camera.StartGrabbing();
        CImageFormatConverter formatConverter;//me
        formatConverter.OutputPixelFormat = PixelType_BGR8packed;//me
        CPylonImage pylonImage;//me

        // This smart pointer will receive the grab result data.
        CGrabResultPtr ptrGrabResult;

        pthread_t thread;
        pthread_create(&thread, NULL, ExposureLoop, NULL);

        // Camera.StopGrabbing() is called automatically by the RetrieveResult() method
        // when c_countOfImagesToGrab images have been retrieved.
        while ( camera.IsGrabbing())
        {
            // Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            camera.RetrieveResult( 5000, ptrGrabResult, TimeoutHandling_ThrowException);

            // Image grabbed successfully?
            if (ptrGrabResult->GrabSucceeded())
            {
                // Access the image data.
                cout << "SizeX: " << ptrGrabResult->GetWidth() << endl;
                cout << "SizeY: " << ptrGrabResult->GetHeight() << endl;
                const uint8_t *pImageBuffer = (uint8_t *) ptrGrabResult->GetBuffer();
                cout << "Gray value of first pixel: " << (uint32_t) pImageBuffer[0] << endl << endl;

                formatConverter.Convert(pylonImage, ptrGrabResult);
                cv::Mat inMat = cv::Mat((int) ptrGrabResult->GetHeight(), (int) ptrGrabResult->GetWidth(), CV_8UC3, (uint8_t *) ptrGrabResult->GetBuffer());
                cv::imshow("image", inMat);
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