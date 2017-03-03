APP_STL := gnustl_static
APP_CPPFLAGS += -fexceptions
APP_CPPFLAGS += -frtti
APP_CFLAGS += -mhard-float -D_NDK_MATH_NO_SOFTFP=1
APP_CPPFLAGS += -mhard-float -D_NDK_MATH_NO_SOFTFP=1
APP_LDFLAGS += -Wl,--no-warn-mismatch -lm_hard
APP_ABI := armeabi-v7a #armeabi x86 arm64-v8a
APP_PLATFORM := android-9
APP_BUILD_SCRIPT := Android.mk
