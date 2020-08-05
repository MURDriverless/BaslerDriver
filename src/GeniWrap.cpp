#include "GeniWrap.hpp"

#include "GeniPylon.hpp"
#include "GeniIDS.hpp"

IGeniCam* IGeniCam::create(GeniImpl geniImpl) {
    switch(geniImpl) {
        case GeniImpl::Pylon_i :
            return new PylonCam();
        case GeniImpl::IDS_i :
            return new IDSCam();
    }

    return nullptr;
}