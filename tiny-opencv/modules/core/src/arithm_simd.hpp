/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_ARITHM_SIMD_HPP__
#define __OPENCV_ARITHM_SIMD_HPP__

namespace cv {

struct NOP {};

#if CV_NEON
#define IF_SIMD(op) op
#else
#define IF_SIMD(op) NOP
#endif


#if CV_NEON

#define FUNCTOR_TEMPLATE(name)          \
    template<typename T> struct name {}

FUNCTOR_TEMPLATE(VLoadStore128);
#if CV_SSE2
FUNCTOR_TEMPLATE(VLoadStore64);
FUNCTOR_TEMPLATE(VLoadStore128Aligned);
#if CV_AVX2
FUNCTOR_TEMPLATE(VLoadStore256);
FUNCTOR_TEMPLATE(VLoadStore256Aligned);
#endif
#endif

#endif

#if CV_NEON

#define FUNCTOR_LOADSTORE(name, template_arg, register_type, load_body, store_body)\
    template <>                                                                \
    struct name<template_arg>{                                                 \
        typedef register_type reg_type;                                        \
        static reg_type load(const template_arg * p) { return load_body (p);}; \
        static void store(template_arg * p, reg_type v) { store_body (p, v);}; \
    }

#define FUNCTOR_CLOSURE_2arg(name, template_arg, body)\
    template<>                                                         \
    struct name<template_arg>                                          \
    {                                                                  \
        VLoadStore128<template_arg>::reg_type operator()(              \
                        VLoadStore128<template_arg>::reg_type a,       \
                        VLoadStore128<template_arg>::reg_type b) const \
        {                                                              \
            return body;                                               \
        };                                                             \
    }

#define FUNCTOR_CLOSURE_1arg(name, template_arg, body)\
    template<>                                                         \
    struct name<template_arg>                                          \
    {                                                                  \
        VLoadStore128<template_arg>::reg_type operator()(              \
                        VLoadStore128<template_arg>::reg_type a,       \
                        VLoadStore128<template_arg>::reg_type  ) const \
        {                                                              \
            return body;                                               \
        };                                                             \
    }

FUNCTOR_LOADSTORE(VLoadStore128,  uchar,  uint8x16_t, vld1q_u8 , vst1q_u8 );
FUNCTOR_LOADSTORE(VLoadStore128,  schar,   int8x16_t, vld1q_s8 , vst1q_s8 );
FUNCTOR_LOADSTORE(VLoadStore128, ushort,  uint16x8_t, vld1q_u16, vst1q_u16);
FUNCTOR_LOADSTORE(VLoadStore128,  short,   int16x8_t, vld1q_s16, vst1q_s16);
FUNCTOR_LOADSTORE(VLoadStore128,    int,   int32x4_t, vld1q_s32, vst1q_s32);
FUNCTOR_LOADSTORE(VLoadStore128,  float, float32x4_t, vld1q_f32, vst1q_f32);

FUNCTOR_TEMPLATE(VAdd);
FUNCTOR_CLOSURE_2arg(VAdd,  uchar, vqaddq_u8 (a, b));
FUNCTOR_CLOSURE_2arg(VAdd,  schar, vqaddq_s8 (a, b));
FUNCTOR_CLOSURE_2arg(VAdd, ushort, vqaddq_u16(a, b));
FUNCTOR_CLOSURE_2arg(VAdd,  short, vqaddq_s16(a, b));
FUNCTOR_CLOSURE_2arg(VAdd,    int, vaddq_s32 (a, b));
FUNCTOR_CLOSURE_2arg(VAdd,  float, vaddq_f32 (a, b));

FUNCTOR_TEMPLATE(VSub);
FUNCTOR_CLOSURE_2arg(VSub,  uchar, vqsubq_u8 (a, b));
FUNCTOR_CLOSURE_2arg(VSub,  schar, vqsubq_s8 (a, b));
FUNCTOR_CLOSURE_2arg(VSub, ushort, vqsubq_u16(a, b));
FUNCTOR_CLOSURE_2arg(VSub,  short, vqsubq_s16(a, b));
FUNCTOR_CLOSURE_2arg(VSub,    int, vsubq_s32 (a, b));
FUNCTOR_CLOSURE_2arg(VSub,  float, vsubq_f32 (a, b));

FUNCTOR_TEMPLATE(VMin);
FUNCTOR_CLOSURE_2arg(VMin,  uchar, vminq_u8 (a, b));
FUNCTOR_CLOSURE_2arg(VMin,  schar, vminq_s8 (a, b));
FUNCTOR_CLOSURE_2arg(VMin, ushort, vminq_u16(a, b));
FUNCTOR_CLOSURE_2arg(VMin,  short, vminq_s16(a, b));
FUNCTOR_CLOSURE_2arg(VMin,    int, vminq_s32(a, b));
FUNCTOR_CLOSURE_2arg(VMin,  float, vminq_f32(a, b));

FUNCTOR_TEMPLATE(VMax);
FUNCTOR_CLOSURE_2arg(VMax,  uchar, vmaxq_u8 (a, b));
FUNCTOR_CLOSURE_2arg(VMax,  schar, vmaxq_s8 (a, b));
FUNCTOR_CLOSURE_2arg(VMax, ushort, vmaxq_u16(a, b));
FUNCTOR_CLOSURE_2arg(VMax,  short, vmaxq_s16(a, b));
FUNCTOR_CLOSURE_2arg(VMax,    int, vmaxq_s32(a, b));
FUNCTOR_CLOSURE_2arg(VMax,  float, vmaxq_f32(a, b));

FUNCTOR_TEMPLATE(VAbsDiff);
FUNCTOR_CLOSURE_2arg(VAbsDiff,  uchar, vabdq_u8  (a, b));
FUNCTOR_CLOSURE_2arg(VAbsDiff,  schar, vqabsq_s8 (vqsubq_s8(a, b)));
FUNCTOR_CLOSURE_2arg(VAbsDiff, ushort, vabdq_u16 (a, b));
FUNCTOR_CLOSURE_2arg(VAbsDiff,  short, vqabsq_s16(vqsubq_s16(a, b)));
FUNCTOR_CLOSURE_2arg(VAbsDiff,    int, vabdq_s32 (a, b));
FUNCTOR_CLOSURE_2arg(VAbsDiff,  float, vabdq_f32 (a, b));

FUNCTOR_TEMPLATE(VAnd);
FUNCTOR_CLOSURE_2arg(VAnd, uchar, vandq_u8(a, b));
FUNCTOR_TEMPLATE(VOr);
FUNCTOR_CLOSURE_2arg(VOr , uchar, vorrq_u8(a, b));
FUNCTOR_TEMPLATE(VXor);
FUNCTOR_CLOSURE_2arg(VXor, uchar, veorq_u8(a, b));
FUNCTOR_TEMPLATE(VNot);
FUNCTOR_CLOSURE_1arg(VNot, uchar, vmvnq_u8(a   ));
#endif


template <typename T>
struct Cmp_SIMD
{
    explicit Cmp_SIMD(int)
    {
    }

    int operator () (const T *, const T *, uchar *, int) const
    {
        return 0;
    }
};

#if CV_NEON

template <>
struct Cmp_SIMD<schar>
{
    explicit Cmp_SIMD(int code_) :
        code(code_)
    {
        // CV_Assert(code == CMP_GT || code == CMP_LE ||
        //           code == CMP_EQ || code == CMP_NE);

        v_mask = vdupq_n_u8(255);
    }

    int operator () (const schar * src1, const schar * src2, uchar * dst, int width) const
    {
        int x = 0;

        if (code == CMP_GT)
            for ( ; x <= width - 16; x += 16)
                vst1q_u8(dst + x, vcgtq_s8(vld1q_s8(src1 + x), vld1q_s8(src2 + x)));
        else if (code == CMP_LE)
            for ( ; x <= width - 16; x += 16)
                vst1q_u8(dst + x, vcleq_s8(vld1q_s8(src1 + x), vld1q_s8(src2 + x)));
        else if (code == CMP_EQ)
            for ( ; x <= width - 16; x += 16)
                vst1q_u8(dst + x, vceqq_s8(vld1q_s8(src1 + x), vld1q_s8(src2 + x)));
        else if (code == CMP_NE)
            for ( ; x <= width - 16; x += 16)
                vst1q_u8(dst + x, veorq_u8(vceqq_s8(vld1q_s8(src1 + x), vld1q_s8(src2 + x)), v_mask));

        return x;
    }

    int code;
    uint8x16_t v_mask;
};

template <>
struct Cmp_SIMD<ushort>
{
    explicit Cmp_SIMD(int code_) :
        code(code_)
    {
        // CV_Assert(code == CMP_GT || code == CMP_LE ||
        //           code == CMP_EQ || code == CMP_NE);

        v_mask = vdup_n_u8(255);
    }

    int operator () (const ushort * src1, const ushort * src2, uchar * dst, int width) const
    {
        int x = 0;

        if (code == CMP_GT)
            for ( ; x <= width - 8; x += 8)
            {
                uint16x8_t v_dst = vcgtq_u16(vld1q_u16(src1 + x), vld1q_u16(src2 + x));
                vst1_u8(dst + x, vmovn_u16(v_dst));
            }
        else if (code == CMP_LE)
            for ( ; x <= width - 8; x += 8)
            {
                uint16x8_t v_dst = vcleq_u16(vld1q_u16(src1 + x), vld1q_u16(src2 + x));
                vst1_u8(dst + x, vmovn_u16(v_dst));
            }
        else if (code == CMP_EQ)
            for ( ; x <= width - 8; x += 8)
            {
                uint16x8_t v_dst = vceqq_u16(vld1q_u16(src1 + x), vld1q_u16(src2 + x));
                vst1_u8(dst + x, vmovn_u16(v_dst));
            }
        else if (code == CMP_NE)
            for ( ; x <= width - 8; x += 8)
            {
                uint16x8_t v_dst = vceqq_u16(vld1q_u16(src1 + x), vld1q_u16(src2 + x));
                vst1_u8(dst + x, veor_u8(vmovn_u16(v_dst), v_mask));
            }

        return x;
    }

    int code;
    uint8x8_t v_mask;
};

template <>
struct Cmp_SIMD<int>
{
    explicit Cmp_SIMD(int code_) :
        code(code_)
    {
        // CV_Assert(code == CMP_GT || code == CMP_LE ||
        //           code == CMP_EQ || code == CMP_NE);

        v_mask = vdup_n_u8(255);
    }

    int operator () (const int * src1, const int * src2, uchar * dst, int width) const
    {
        int x = 0;

        if (code == CMP_GT)
            for ( ; x <= width - 8; x += 8)
            {
                uint32x4_t v_dst1 = vcgtq_s32(vld1q_s32(src1 + x), vld1q_s32(src2 + x));
                uint32x4_t v_dst2 = vcgtq_s32(vld1q_s32(src1 + x + 4), vld1q_s32(src2 + x + 4));
                vst1_u8(dst + x, vmovn_u16(vcombine_u16(vmovn_u32(v_dst1), vmovn_u32(v_dst2))));
            }
        else if (code == CMP_LE)
            for ( ; x <= width - 8; x += 8)
            {
                uint32x4_t v_dst1 = vcleq_s32(vld1q_s32(src1 + x), vld1q_s32(src2 + x));
                uint32x4_t v_dst2 = vcleq_s32(vld1q_s32(src1 + x + 4), vld1q_s32(src2 + x + 4));
                vst1_u8(dst + x, vmovn_u16(vcombine_u16(vmovn_u32(v_dst1), vmovn_u32(v_dst2))));
            }
        else if (code == CMP_EQ)
            for ( ; x <= width - 8; x += 8)
            {
                uint32x4_t v_dst1 = vceqq_s32(vld1q_s32(src1 + x), vld1q_s32(src2 + x));
                uint32x4_t v_dst2 = vceqq_s32(vld1q_s32(src1 + x + 4), vld1q_s32(src2 + x + 4));
                vst1_u8(dst + x, vmovn_u16(vcombine_u16(vmovn_u32(v_dst1), vmovn_u32(v_dst2))));
            }
        else if (code == CMP_NE)
            for ( ; x <= width - 8; x += 8)
            {
                uint32x4_t v_dst1 = vceqq_s32(vld1q_s32(src1 + x), vld1q_s32(src2 + x));
                uint32x4_t v_dst2 = vceqq_s32(vld1q_s32(src1 + x + 4), vld1q_s32(src2 + x + 4));
                uint8x8_t v_dst = vmovn_u16(vcombine_u16(vmovn_u32(v_dst1), vmovn_u32(v_dst2)));
                vst1_u8(dst + x, veor_u8(v_dst, v_mask));
            }

        return x;
    }

    int code;
    uint8x8_t v_mask;
};

template <>
struct Cmp_SIMD<float>
{
    explicit Cmp_SIMD(int code_) :
        code(code_)
    {
        // CV_Assert(code == CMP_GT || code == CMP_LE ||
        //           code == CMP_EQ || code == CMP_NE);

        v_mask = vdup_n_u8(255);
    }

    int operator () (const float * src1, const float * src2, uchar * dst, int width) const
    {
        int x = 0;

        if (code == CMP_GT)
            for ( ; x <= width - 8; x += 8)
            {
                uint32x4_t v_dst1 = vcgtq_f32(vld1q_f32(src1 + x), vld1q_f32(src2 + x));
                uint32x4_t v_dst2 = vcgtq_f32(vld1q_f32(src1 + x + 4), vld1q_f32(src2 + x + 4));
                vst1_u8(dst + x, vmovn_u16(vcombine_u16(vmovn_u32(v_dst1), vmovn_u32(v_dst2))));
            }
        else if (code == CMP_LE)
            for ( ; x <= width - 8; x += 8)
            {
                uint32x4_t v_dst1 = vcleq_f32(vld1q_f32(src1 + x), vld1q_f32(src2 + x));
                uint32x4_t v_dst2 = vcleq_f32(vld1q_f32(src1 + x + 4), vld1q_f32(src2 + x + 4));
                vst1_u8(dst + x, vmovn_u16(vcombine_u16(vmovn_u32(v_dst1), vmovn_u32(v_dst2))));
            }
        else if (code == CMP_EQ)
            for ( ; x <= width - 8; x += 8)
            {
                uint32x4_t v_dst1 = vceqq_f32(vld1q_f32(src1 + x), vld1q_f32(src2 + x));
                uint32x4_t v_dst2 = vceqq_f32(vld1q_f32(src1 + x + 4), vld1q_f32(src2 + x + 4));
                vst1_u8(dst + x, vmovn_u16(vcombine_u16(vmovn_u32(v_dst1), vmovn_u32(v_dst2))));
            }
        else if (code == CMP_NE)
            for ( ; x <= width - 8; x += 8)
            {
                uint32x4_t v_dst1 = vceqq_f32(vld1q_f32(src1 + x), vld1q_f32(src2 + x));
                uint32x4_t v_dst2 = vceqq_f32(vld1q_f32(src1 + x + 4), vld1q_f32(src2 + x + 4));
                uint8x8_t v_dst = vmovn_u16(vcombine_u16(vmovn_u32(v_dst1), vmovn_u32(v_dst2)));
                vst1_u8(dst + x, veor_u8(v_dst, v_mask));
            }

        return x;
    }

    int code;
    uint8x8_t v_mask;
};

#endif


template <typename T, typename WT>
struct Mul_SIMD
{
    int operator() (const T *, const T *, T *, int, WT) const
    {
        return 0;
    }
};

#if CV_NEON

template <>
struct Mul_SIMD<uchar, float>
{
    int operator() (const uchar * src1, const uchar * src2, uchar * dst, int width, float scale) const
    {
        int x = 0;

        if( scale == 1.0f )
            for ( ; x <= width - 8; x += 8)
            {
                uint16x8_t v_src1 = vmovl_u8(vld1_u8(src1 + x));
                uint16x8_t v_src2 = vmovl_u8(vld1_u8(src2 + x));

                float32x4_t v_dst1 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1))),
                                               vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src2))));
                float32x4_t v_dst2 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1))),
                                               vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src2))));

                uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)),
                                                vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
                vst1_u8(dst + x, vqmovn_u16(v_dst));
            }
        else
        {
            float32x4_t v_scale = vdupq_n_f32(scale);
            for ( ; x <= width - 8; x += 8)
            {
                uint16x8_t v_src1 = vmovl_u8(vld1_u8(src1 + x));
                uint16x8_t v_src2 = vmovl_u8(vld1_u8(src2 + x));

                float32x4_t v_dst1 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1))),
                                               vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src2))));
                v_dst1 = vmulq_f32(v_dst1, v_scale);
                float32x4_t v_dst2 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1))),
                                               vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src2))));
                v_dst2 = vmulq_f32(v_dst2, v_scale);

                uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)),
                                                vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
                vst1_u8(dst + x, vqmovn_u16(v_dst));
            }
        }

        return x;
    }
};

template <>
struct Mul_SIMD<schar, float>
{
    int operator() (const schar * src1, const schar * src2, schar * dst, int width, float scale) const
    {
        int x = 0;

        if( scale == 1.0f )
            for ( ; x <= width - 8; x += 8)
            {
                int16x8_t v_src1 = vmovl_s8(vld1_s8(src1 + x));
                int16x8_t v_src2 = vmovl_s8(vld1_s8(src2 + x));

                float32x4_t v_dst1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src1))),
                                               vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src2))));
                float32x4_t v_dst2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src1))),
                                               vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src2))));

                int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)),
                                               vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
                vst1_s8(dst + x, vqmovn_s16(v_dst));
            }
        else
        {
            float32x4_t v_scale = vdupq_n_f32(scale);
            for ( ; x <= width - 8; x += 8)
            {
                int16x8_t v_src1 = vmovl_s8(vld1_s8(src1 + x));
                int16x8_t v_src2 = vmovl_s8(vld1_s8(src2 + x));

                float32x4_t v_dst1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src1))),
                                               vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src2))));
                v_dst1 = vmulq_f32(v_dst1, v_scale);
                float32x4_t v_dst2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src1))),
                                               vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src2))));
                v_dst2 = vmulq_f32(v_dst2, v_scale);

                int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)),
                                               vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
                vst1_s8(dst + x, vqmovn_s16(v_dst));
            }
        }

        return x;
    }
};

template <>
struct Mul_SIMD<ushort, float>
{
    int operator() (const ushort * src1, const ushort * src2, ushort * dst, int width, float scale) const
    {
        int x = 0;

        if( scale == 1.0f )
            for ( ; x <= width - 8; x += 8)
            {
                uint16x8_t v_src1 = vld1q_u16(src1 + x), v_src2 = vld1q_u16(src2 + x);

                float32x4_t v_dst1 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1))),
                                               vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src2))));
                float32x4_t v_dst2 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1))),
                                               vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src2))));

                uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)),
                                                vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
                vst1q_u16(dst + x, v_dst);
            }
        else
        {
            float32x4_t v_scale = vdupq_n_f32(scale);
            for ( ; x <= width - 8; x += 8)
            {
                uint16x8_t v_src1 = vld1q_u16(src1 + x), v_src2 = vld1q_u16(src2 + x);

                float32x4_t v_dst1 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1))),
                                               vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src2))));
                v_dst1 = vmulq_f32(v_dst1, v_scale);
                float32x4_t v_dst2 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1))),
                                               vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src2))));
                v_dst2 = vmulq_f32(v_dst2, v_scale);

                uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)),
                                                vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
                vst1q_u16(dst + x, v_dst);
            }
        }

        return x;
    }
};

template <>
struct Mul_SIMD<short, float>
{
    int operator() (const short * src1, const short * src2, short * dst, int width, float scale) const
    {
        int x = 0;

        if( scale == 1.0f )
            for ( ; x <= width - 8; x += 8)
            {
                int16x8_t v_src1 = vld1q_s16(src1 + x), v_src2 = vld1q_s16(src2 + x);

                float32x4_t v_dst1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src1))),
                                               vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src2))));
                float32x4_t v_dst2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src1))),
                                               vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src2))));

                int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)),
                                               vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
                vst1q_s16(dst + x, v_dst);
            }
        else
        {
            float32x4_t v_scale = vdupq_n_f32(scale);
            for ( ; x <= width - 8; x += 8)
            {
                int16x8_t v_src1 = vld1q_s16(src1 + x), v_src2 = vld1q_s16(src2 + x);

                float32x4_t v_dst1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src1))),
                                               vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src2))));
                v_dst1 = vmulq_f32(v_dst1, v_scale);
                float32x4_t v_dst2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src1))),
                                               vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src2))));
                v_dst2 = vmulq_f32(v_dst2, v_scale);

                int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)),
                                               vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
                vst1q_s16(dst + x, v_dst);
            }
        }

        return x;
    }
};

template <>
struct Mul_SIMD<float, float>
{
    int operator() (const float * src1, const float * src2, float * dst, int width, float scale) const
    {
        int x = 0;

        if( scale == 1.0f )
            for ( ; x <= width - 8; x += 8)
            {
                float32x4_t v_dst1 = vmulq_f32(vld1q_f32(src1 + x), vld1q_f32(src2 + x));
                float32x4_t v_dst2 = vmulq_f32(vld1q_f32(src1 + x + 4), vld1q_f32(src2 + x + 4));
                vst1q_f32(dst + x, v_dst1);
                vst1q_f32(dst + x + 4, v_dst2);
            }
        else
        {
            float32x4_t v_scale = vdupq_n_f32(scale);
            for ( ; x <= width - 8; x += 8)
            {
                float32x4_t v_dst1 = vmulq_f32(vld1q_f32(src1 + x), vld1q_f32(src2 + x));
                v_dst1 = vmulq_f32(v_dst1, v_scale);

                float32x4_t v_dst2 = vmulq_f32(vld1q_f32(src1 + x + 4), vld1q_f32(src2 + x + 4));
                v_dst2 = vmulq_f32(v_dst2, v_scale);

                vst1q_f32(dst + x, v_dst1);
                vst1q_f32(dst + x + 4, v_dst2);
            }
        }

        return x;
    }
};

#endif

template <typename T>
struct Div_SIMD
{
    int operator() (const T *, const T *, T *, int, double) const
    {
        return 0;
    }
};

template <typename T>
struct Recip_SIMD
{
    int operator() (const T *, T *, int, double) const
    {
        return 0;
    }
};

template <typename T, typename WT>
struct AddWeighted_SIMD
{
    int operator() (const T *, const T *, T *, int, WT, WT, WT) const
    {
        return 0;
    }
};

}

#endif // __OPENCV_ARITHM_SIMD_HPP__
