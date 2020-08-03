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
#include <vector>
#include <algorithm>


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

    auto& deviceManager = peak::DeviceManager::Instance();
    deviceManager.AddProducerLibrary("/usr/lib/ids/cti/ids_u3vgentl.cti");
    deviceManager.Update(peak::DeviceManager::UpdatePolicy::DontScanEnvironmentForProducerLibraries);
    auto devices = deviceManager.Devices();

    for (auto device : devices) {
        cout << device->DisplayName() << endl;
    }

    auto device = devices.at(0)->OpenDevice(peak::core::DeviceAccessType::Control);
    auto nodeMapRemoteDevice = device->RemoteDevice()->NodeMaps().at(0);
    
    auto dataStream = device->DataStreams().at(0)->OpenDataStream();
    // get payload size
    auto payloadSize = nodeMapRemoteDevice->FindNode<peak::core::nodes::IntegerNode>("PayloadSize")->Value();

    nodeMapRemoteDevice->FindNode<peak::core::nodes::EnumerationNode>("PixelFormat")->SetCurrentEntry("Mono8");
    
    // get number of buffers to allocate
    // the buffer count depends on your application, here the minimum required number for the data stream
    auto bufferCountMax = dataStream->NumBuffersAnnouncedMinRequired();
    
    // allocate and announce image buffers and queue them
    for (uint64_t bufferCount = 0; bufferCount < bufferCountMax; ++bufferCount)
    {
        auto buffer = dataStream->AllocAndAnnounceBuffer(static_cast<size_t>(payloadSize), nullptr);
        dataStream->QueueBuffer(buffer);
    }

    dataStream->StartAcquisition(peak::core::AcquisitionStartMode::Default);
    
    // start the device
    nodeMapRemoteDevice->FindNode<peak::core::nodes::CommandNode>("AcquisitionStart")->Execute();
    
    // the acquisition loop
    bool is_running = true;
    while (is_running)
    {
        // get buffer from data stream and process it
        auto buffer = dataStream->WaitForFinishedBuffer(5000);
        
        size_t width = buffer->Width();
        size_t height = buffer->Height();
        cv::Mat mat(height, width, CV_8UC1, static_cast<u_int8_t*>(buffer->BasePtr()));

        cv::imshow("ehh", mat);
        cv::waitKey(1);

        dataStream->QueueBuffer(buffer);
    }

    // stop the data stream
    dataStream->StopAcquisition(peak::core::AcquisitionStopMode::Default);
    
    // stop the device
    nodeMapRemoteDevice->FindNode<peak::core::nodes::CommandNode>("AcquisitionStop")->Execute();
    
    // flush all buffers
    dataStream->Flush(peak::core::DataStreamFlushMode::DiscardAll);
    
    // revoke all buffers
    for (const auto& buffer : dataStream->AnnouncedBuffers())
    {
        dataStream->RevokeBuffer(buffer);
    }

    peak::Library::Close();

    return exitCode;
}