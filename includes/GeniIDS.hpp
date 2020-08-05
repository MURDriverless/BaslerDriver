#pragma once
#include "GeniWrap.hpp"

#include <peak/peak.hpp>

class IDSCam: public IGeniCam {
    private:
        std::shared_ptr<peak::core::DataStream> dataStream;
        std::shared_ptr<peak::core::Device> device;
        std::shared_ptr<peak::core::NodeMap> nodeMapRemoteDevice;
        int64_t payloadSize;

        std::shared_ptr<peak::core::Buffer> _buffer;
    public:
        void initializeLibrary() {
            peak::Library::Initialize();
        }

        void finalizeLibrary() {
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
        }

        void setup(const std::string cameraName) {
            auto& deviceManager = peak::DeviceManager::Instance();
            deviceManager.AddProducerLibrary("/usr/lib/ids/cti/ids_u3vgentl.cti");
            deviceManager.Update(peak::DeviceManager::UpdatePolicy::DontScanEnvironmentForProducerLibraries);
            auto devices = deviceManager.Devices();

            device = devices.at(0)->OpenDevice(peak::core::DeviceAccessType::Control);
            nodeMapRemoteDevice = device->RemoteDevice()->NodeMaps().at(0);
            
            dataStream = device->DataStreams().at(0)->OpenDataStream();
            // get payload size
            payloadSize = nodeMapRemoteDevice->FindNode<peak::core::nodes::IntegerNode>("PayloadSize")->Value();

            nodeMapRemoteDevice->FindNode<peak::core::nodes::EnumerationNode>("PixelFormat")->SetCurrentEntry("BayerBG8");
        }

        void startGrabbing(uint32_t numImages = UINT32_MAX) {
            // get number of buffers to allocate
            // the buffer count depends on your application, here the minimum required number for the data stream
            auto bufferCountMax = dataStream->NumBuffersAnnouncedMinRequired();
            
            // allocate and announce image buffers and queue them
            for (uint64_t bufferCount = 0; bufferCount < bufferCountMax; ++bufferCount)
            {
                auto buffer = dataStream->AllocAndAnnounceBuffer(static_cast<size_t>(payloadSize), nullptr);
                dataStream->QueueBuffer(buffer);
            }
            dataStream->StartAcquisition(peak::core::AcquisitionStartMode::Default, numImages);
            
            // start the device
            nodeMapRemoteDevice->FindNode<peak::core::nodes::CommandNode>("AcquisitionStart")->Execute();
        }

        bool isGrabbing() {
            return dataStream->IsGrabbing();
        }

        bool retreiveResult(int &height, int &width, uint8_t* &buffer) {
            // get buffer from data stream and process it
            _buffer = dataStream->WaitForFinishedBuffer(5000);

            height = _buffer->Height();
            width  = _buffer->Width();
            buffer = static_cast<uint8_t*>(_buffer->BasePtr());

            return true;
        }

        void clearResult() {
            dataStream->QueueBuffer(_buffer);
        }
};