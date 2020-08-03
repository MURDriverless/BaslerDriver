#pragma once
#include <iostream>

class IGeniCam {
    public:
        virtual void initializeLibrary();
        virtual void finalizeLibrary();
        virtual void setup(const std::string cameraName);
        virtual void startGrabbing(uint32_t numImages = UINT32_MAX);
        virtual bool isGrabbing();
        virtual bool retreiveResult(int &height, int &width, uint8_t* &buffer);
        virtual void clearResult();
};