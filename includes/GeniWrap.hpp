#pragma once
#include <iostream>

class IGeniCam {
    public:
        virtual void initializeLibrary() = 0;
        virtual void finalizeLibrary() = 0;
        virtual void setup(const std::string cameraName) = 0;
        virtual void startGrabbing(uint32_t numImages = UINT32_MAX) = 0;
        virtual bool isGrabbing() = 0;
        virtual bool retreiveResult(int &height, int &width, uint8_t* &buffer) = 0;
        virtual void clearResult() = 0;
};