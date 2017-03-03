LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

LOCAL_MODULE := opencv

LOCAL_C_INCLUDES := $(LOCAL_PATH)/include \
                    $(LOCAL_PATH)/modules/core/include \
                    $(LOCAL_PATH)/modules/imgproc/include \
                    $(LOCAL_PATH)/modules/imgcodecs/include \
                    $(LOCAL_PATH)/modules/calib3d/include

# opencv_core
OPENCV_CORE_CPP_LIST := $(wildcard $(LOCAL_PATH)/modules/core/src/*.cpp)
LOCAL_SRC_FILES += $(OPENCV_CORE_CPP_LIST:$(LOCAL_PATH)/%=%)

# imgproc
OPENCV_IMGPROC_CPP_LIST := $(wildcard $(LOCAL_PATH)/modules/imgproc/src/*.cpp)
LOCAL_SRC_FILES += $(OPENCV_IMGPROC_CPP_LIST:$(LOCAL_PATH)/%=%)

# imgcodecs
OPENCV_IMGCODECS_CPP_LIST := $(wildcard $(LOCAL_PATH)/modules/imgcodecs/src/*.cpp)
LOCAL_SRC_FILES += $(OPENCV_IMGCODECS_CPP_LIST:$(LOCAL_PATH)/%=%)

# calib3d
OPENCV_CALIB3D_CPP_LIST := $(wildcard $(LOCAL_PATH)/modules/calib3d/src/*.cpp)
LOCAL_SRC_FILES += $(OPENCV_CALIB3D_CPP_LIST:$(LOCAL_PATH)/%=%)

LOCAL_CFLAGS += -O3 -fPIC -Wall -Wno-unused -Wno-sign-compare -Wno-deprecated-declarations -Wmaybe-uninitialized
LOCAL_CPPFLAGS += -O3 -std=c++11 -mfloat-abi=softfp -mfpu=neon -D__OPENCV_BUILD=1 -Wmaybe-uninitialized
LOCAL_LDLIBS += -llog -lz

include $(BUILD_SHARED_LIBRARY)
