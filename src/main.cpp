//IDS Camera Headers
#include <peak/peak.hpp>

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

    peak::Library::Initialize();

    auto ehh = peak::core::EnvironmentInspector::CollectCTIPaths();

    for (auto ehh2 : ehh) {
        cout << ehh2 << endl;
    }

    auto& deviceManager = peak::DeviceManager::Instance();
    deviceManager.Update();
    auto devices = deviceManager.Devices();

    for (auto device : devices) {
        cout << device->DisplayName() << endl;
    }

    peak::Library::Close();

    return exitCode;
}