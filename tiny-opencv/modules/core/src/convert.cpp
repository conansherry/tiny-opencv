/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
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

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

#define CV_SPLIT_MERGE_MAX_BLOCK_SIZE(cn) ((INT_MAX/4)/cn) // HAL implementation accepts 'int' len, so INT_MAX doesn't work here

/****************************************************************************************\
*                                       split & merge                                    *
\****************************************************************************************/

typedef void (*SplitFunc)(const uchar* src, uchar** dst, int len, int cn);

static SplitFunc getSplitFunc(int depth)
{
    static SplitFunc splitTab[] =
    {
        (SplitFunc)GET_OPTIMIZED(cv::hal::split8u), (SplitFunc)GET_OPTIMIZED(cv::hal::split8u), (SplitFunc)GET_OPTIMIZED(cv::hal::split16u), (SplitFunc)GET_OPTIMIZED(cv::hal::split16u),
        (SplitFunc)GET_OPTIMIZED(cv::hal::split32s), (SplitFunc)GET_OPTIMIZED(cv::hal::split32s), (SplitFunc)GET_OPTIMIZED(cv::hal::split64s), 0
    };

    return splitTab[depth];
}

typedef void (*MergeFunc)(const uchar** src, uchar* dst, int len, int cn);

static MergeFunc getMergeFunc(int depth)
{
    static MergeFunc mergeTab[] =
    {
        (MergeFunc)GET_OPTIMIZED(cv::hal::merge8u), (MergeFunc)GET_OPTIMIZED(cv::hal::merge8u), (MergeFunc)GET_OPTIMIZED(cv::hal::merge16u), (MergeFunc)GET_OPTIMIZED(cv::hal::merge16u),
        (MergeFunc)GET_OPTIMIZED(cv::hal::merge32s), (MergeFunc)GET_OPTIMIZED(cv::hal::merge32s), (MergeFunc)GET_OPTIMIZED(cv::hal::merge64s), 0
    };

    return mergeTab[depth];
}

void cv::split(const Mat& src, Mat* mv)
{
    CV_INSTRUMENT_REGION()

    int k, depth = src.depth(), cn = src.channels();
    if( cn == 1 )
    {
        src.copyTo(mv[0]);
        return;
    }

    SplitFunc func = getSplitFunc(depth);
    CV_Assert( func != 0 );

    size_t esz = src.elemSize(), esz1 = src.elemSize1();
    size_t blocksize0 = (BLOCK_SIZE + esz-1)/esz;
    AutoBuffer<uchar> _buf((cn+1)*(sizeof(Mat*) + sizeof(uchar*)) + 16);
    const Mat** arrays = (const Mat**)(uchar*)_buf;
    uchar** ptrs = (uchar**)alignPtr(arrays + cn + 1, 16);

    arrays[0] = &src;
    for( k = 0; k < cn; k++ )
    {
        mv[k].create(src.dims, src.size, depth);
        arrays[k+1] = &mv[k];
    }

    NAryMatIterator it(arrays, ptrs, cn+1);
    size_t total = it.size;
    size_t blocksize = std::min((size_t)CV_SPLIT_MERGE_MAX_BLOCK_SIZE(cn), cn <= 4 ? total : std::min(total, blocksize0));

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( size_t j = 0; j < total; j += blocksize )
        {
            size_t bsz = std::min(total - j, blocksize);
            func( ptrs[0], &ptrs[1], (int)bsz, cn );

            if( j + blocksize < total )
            {
                ptrs[0] += bsz*esz;
                for( k = 0; k < cn; k++ )
                    ptrs[k+1] += bsz*esz1;
            }
        }
    }
}


void cv::split(InputArray _m, OutputArrayOfArrays _mv)
{
    CV_INSTRUMENT_REGION()

    Mat m = _m.getMat();
    if( m.empty() )
    {
        _mv.release();
        return;
    }

    CV_Assert( !_mv.fixedType() || _mv.empty() || _mv.type() == m.depth() );

    int depth = m.depth(), cn = m.channels();
    _mv.create(cn, 1, depth);
    for (int i = 0; i < cn; ++i)
        _mv.create(m.dims, m.size.p, depth, i);

    std::vector<Mat> dst;
    _mv.getMatVector(dst);

    split(m, &dst[0]);
}

void cv::merge(const Mat* mv, size_t n, OutputArray _dst)
{
    CV_INSTRUMENT_REGION()

    CV_Assert( mv && n > 0 );

    int depth = mv[0].depth();
    bool allch1 = true;
    int k, cn = 0;
    size_t i;

    for( i = 0; i < n; i++ )
    {
        CV_Assert(mv[i].size == mv[0].size && mv[i].depth() == depth);
        allch1 = allch1 && mv[i].channels() == 1;
        cn += mv[i].channels();
    }

    CV_Assert( 0 < cn && cn <= CV_CN_MAX );
    _dst.create(mv[0].dims, mv[0].size, CV_MAKETYPE(depth, cn));
    Mat dst = _dst.getMat();

    if( n == 1 )
    {
        mv[0].copyTo(dst);
        return;
    }

    if( !allch1 )
    {
        AutoBuffer<int> pairs(cn*2);
        int j, ni=0;

        for( i = 0, j = 0; i < n; i++, j += ni )
        {
            ni = mv[i].channels();
            for( k = 0; k < ni; k++ )
            {
                pairs[(j+k)*2] = j + k;
                pairs[(j+k)*2+1] = j + k;
            }
        }
        mixChannels( mv, n, &dst, 1, &pairs[0], cn );
        return;
    }

    MergeFunc func = getMergeFunc(depth);
    CV_Assert( func != 0 );

    size_t esz = dst.elemSize(), esz1 = dst.elemSize1();
    size_t blocksize0 = (int)((BLOCK_SIZE + esz-1)/esz);
    AutoBuffer<uchar> _buf((cn+1)*(sizeof(Mat*) + sizeof(uchar*)) + 16);
    const Mat** arrays = (const Mat**)(uchar*)_buf;
    uchar** ptrs = (uchar**)alignPtr(arrays + cn + 1, 16);

    arrays[0] = &dst;
    for( k = 0; k < cn; k++ )
        arrays[k+1] = &mv[k];

    NAryMatIterator it(arrays, ptrs, cn+1);
    size_t total = (int)it.size;
    size_t blocksize = std::min((size_t)CV_SPLIT_MERGE_MAX_BLOCK_SIZE(cn), cn <= 4 ? total : std::min(total, blocksize0));

    for( i = 0; i < it.nplanes; i++, ++it )
    {
        for( size_t j = 0; j < total; j += blocksize )
        {
            size_t bsz = std::min(total - j, blocksize);
            func( (const uchar**)&ptrs[1], ptrs[0], (int)bsz, cn );

            if( j + blocksize < total )
            {
                ptrs[0] += bsz*esz;
                for( int t = 0; t < cn; t++ )
                    ptrs[t+1] += bsz*esz1;
            }
        }
    }
}


void cv::merge(InputArrayOfArrays _mv, OutputArray _dst)
{
    CV_INSTRUMENT_REGION()

    std::vector<Mat> mv;
    _mv.getMatVector(mv);
    merge(!mv.empty() ? &mv[0] : 0, mv.size(), _dst);
}

/****************************************************************************************\
*                       Generalized split/merge: mixing channels                         *
\****************************************************************************************/

namespace cv
{

template<typename T> static void
mixChannels_( const T** src, const int* sdelta,
              T** dst, const int* ddelta,
              int len, int npairs )
{
    int i, k;
    for( k = 0; k < npairs; k++ )
    {
        const T* s = src[k];
        T* d = dst[k];
        int ds = sdelta[k], dd = ddelta[k];
        if( s )
        {
            for( i = 0; i <= len - 2; i += 2, s += ds*2, d += dd*2 )
            {
                T t0 = s[0], t1 = s[ds];
                d[0] = t0; d[dd] = t1;
            }
            if( i < len )
                d[0] = s[0];
        }
        else
        {
            for( i = 0; i <= len - 2; i += 2, d += dd*2 )
                d[0] = d[dd] = 0;
            if( i < len )
                d[0] = 0;
        }
    }
}


static void mixChannels8u( const uchar** src, const int* sdelta,
                           uchar** dst, const int* ddelta,
                           int len, int npairs )
{
    mixChannels_(src, sdelta, dst, ddelta, len, npairs);
}

static void mixChannels16u( const ushort** src, const int* sdelta,
                            ushort** dst, const int* ddelta,
                            int len, int npairs )
{
    mixChannels_(src, sdelta, dst, ddelta, len, npairs);
}

static void mixChannels32s( const int** src, const int* sdelta,
                            int** dst, const int* ddelta,
                            int len, int npairs )
{
    mixChannels_(src, sdelta, dst, ddelta, len, npairs);
}

static void mixChannels64s( const int64** src, const int* sdelta,
                            int64** dst, const int* ddelta,
                            int len, int npairs )
{
    mixChannels_(src, sdelta, dst, ddelta, len, npairs);
}

typedef void (*MixChannelsFunc)( const uchar** src, const int* sdelta,
        uchar** dst, const int* ddelta, int len, int npairs );

static MixChannelsFunc getMixchFunc(int depth)
{
    static MixChannelsFunc mixchTab[] =
    {
        (MixChannelsFunc)mixChannels8u, (MixChannelsFunc)mixChannels8u, (MixChannelsFunc)mixChannels16u,
        (MixChannelsFunc)mixChannels16u, (MixChannelsFunc)mixChannels32s, (MixChannelsFunc)mixChannels32s,
        (MixChannelsFunc)mixChannels64s, 0
    };

    return mixchTab[depth];
}

}

void cv::mixChannels( const Mat* src, size_t nsrcs, Mat* dst, size_t ndsts, const int* fromTo, size_t npairs )
{
    CV_INSTRUMENT_REGION()

    if( npairs == 0 )
        return;
    CV_Assert( src && nsrcs > 0 && dst && ndsts > 0 && fromTo && npairs > 0 );

    size_t i, j, k, esz1 = dst[0].elemSize1();
    int depth = dst[0].depth();

    AutoBuffer<uchar> buf((nsrcs + ndsts + 1)*(sizeof(Mat*) + sizeof(uchar*)) + npairs*(sizeof(uchar*)*2 + sizeof(int)*6));
    const Mat** arrays = (const Mat**)(uchar*)buf;
    uchar** ptrs = (uchar**)(arrays + nsrcs + ndsts);
    const uchar** srcs = (const uchar**)(ptrs + nsrcs + ndsts + 1);
    uchar** dsts = (uchar**)(srcs + npairs);
    int* tab = (int*)(dsts + npairs);
    int *sdelta = (int*)(tab + npairs*4), *ddelta = sdelta + npairs;

    for( i = 0; i < nsrcs; i++ )
        arrays[i] = &src[i];
    for( i = 0; i < ndsts; i++ )
        arrays[i + nsrcs] = &dst[i];
    ptrs[nsrcs + ndsts] = 0;

    for( i = 0; i < npairs; i++ )
    {
        int i0 = fromTo[i*2], i1 = fromTo[i*2+1];
        if( i0 >= 0 )
        {
            for( j = 0; j < nsrcs; i0 -= src[j].channels(), j++ )
                if( i0 < src[j].channels() )
                    break;
            CV_Assert(j < nsrcs && src[j].depth() == depth);
            tab[i*4] = (int)j; tab[i*4+1] = (int)(i0*esz1);
            sdelta[i] = src[j].channels();
        }
        else
        {
            tab[i*4] = (int)(nsrcs + ndsts); tab[i*4+1] = 0;
            sdelta[i] = 0;
        }

        for( j = 0; j < ndsts; i1 -= dst[j].channels(), j++ )
            if( i1 < dst[j].channels() )
                break;
        CV_Assert(i1 >= 0 && j < ndsts && dst[j].depth() == depth);
        tab[i*4+2] = (int)(j + nsrcs); tab[i*4+3] = (int)(i1*esz1);
        ddelta[i] = dst[j].channels();
    }

    NAryMatIterator it(arrays, ptrs, (int)(nsrcs + ndsts));
    int total = (int)it.size, blocksize = std::min(total, (int)((BLOCK_SIZE + esz1-1)/esz1));
    MixChannelsFunc func = getMixchFunc(depth);

    for( i = 0; i < it.nplanes; i++, ++it )
    {
        for( k = 0; k < npairs; k++ )
        {
            srcs[k] = ptrs[tab[k*4]] + tab[k*4+1];
            dsts[k] = ptrs[tab[k*4+2]] + tab[k*4+3];
        }

        for( int t = 0; t < total; t += blocksize )
        {
            int bsz = std::min(total - t, blocksize);
            func( srcs, sdelta, dsts, ddelta, bsz, (int)npairs );

            if( t + blocksize < total )
                for( k = 0; k < npairs; k++ )
                {
                    srcs[k] += blocksize*sdelta[k]*esz1;
                    dsts[k] += blocksize*ddelta[k]*esz1;
                }
        }
    }
}

void cv::mixChannels(InputArrayOfArrays src, InputOutputArrayOfArrays dst,
                 const int* fromTo, size_t npairs)
{
    CV_INSTRUMENT_REGION()

    if (npairs == 0 || fromTo == NULL)
        return;

    bool src_is_mat = src.kind() != _InputArray::STD_VECTOR_MAT &&
            src.kind() != _InputArray::STD_VECTOR_VECTOR &&
            src.kind() != _InputArray::STD_VECTOR_UMAT;
    bool dst_is_mat = dst.kind() != _InputArray::STD_VECTOR_MAT &&
            dst.kind() != _InputArray::STD_VECTOR_VECTOR &&
            dst.kind() != _InputArray::STD_VECTOR_UMAT;
    int i;
    int nsrc = src_is_mat ? 1 : (int)src.total();
    int ndst = dst_is_mat ? 1 : (int)dst.total();

    CV_Assert(nsrc > 0 && ndst > 0);
    cv::AutoBuffer<Mat> _buf(nsrc + ndst);
    Mat* buf = _buf;
    for( i = 0; i < nsrc; i++ )
        buf[i] = src.getMat(src_is_mat ? -1 : i);
    for( i = 0; i < ndst; i++ )
        buf[nsrc + i] = dst.getMat(dst_is_mat ? -1 : i);
    mixChannels(&buf[0], nsrc, &buf[nsrc], ndst, fromTo, npairs);
}

void cv::mixChannels(InputArrayOfArrays src, InputOutputArrayOfArrays dst,
                     const std::vector<int>& fromTo)
{
    CV_INSTRUMENT_REGION()

    if (fromTo.empty())
        return;

    bool src_is_mat = src.kind() != _InputArray::STD_VECTOR_MAT &&
            src.kind() != _InputArray::STD_VECTOR_VECTOR &&
            src.kind() != _InputArray::STD_VECTOR_UMAT;
    bool dst_is_mat = dst.kind() != _InputArray::STD_VECTOR_MAT &&
            dst.kind() != _InputArray::STD_VECTOR_VECTOR &&
            dst.kind() != _InputArray::STD_VECTOR_UMAT;
    int i;
    int nsrc = src_is_mat ? 1 : (int)src.total();
    int ndst = dst_is_mat ? 1 : (int)dst.total();

    CV_Assert(fromTo.size()%2 == 0 && nsrc > 0 && ndst > 0);
    cv::AutoBuffer<Mat> _buf(nsrc + ndst);
    Mat* buf = _buf;
    for( i = 0; i < nsrc; i++ )
        buf[i] = src.getMat(src_is_mat ? -1 : i);
    for( i = 0; i < ndst; i++ )
        buf[nsrc + i] = dst.getMat(dst_is_mat ? -1 : i);
    mixChannels(&buf[0], nsrc, &buf[nsrc], ndst, &fromTo[0], fromTo.size()/2);
}

void cv::extractChannel(InputArray _src, OutputArray _dst, int coi)
{
    CV_INSTRUMENT_REGION()

    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert( 0 <= coi && coi < cn );
    int ch[] = { coi, 0 };

    Mat src = _src.getMat();
    _dst.create(src.dims, &src.size[0], depth);
    Mat dst = _dst.getMat();
    mixChannels(&src, 1, &dst, 1, ch, 1);
}

void cv::insertChannel(InputArray _src, InputOutputArray _dst, int coi)
{
    CV_INSTRUMENT_REGION()

    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), scn = CV_MAT_CN(stype);
    int dtype = _dst.type(), ddepth = CV_MAT_DEPTH(dtype), dcn = CV_MAT_CN(dtype);
    CV_Assert( _src.sameSize(_dst) && sdepth == ddepth );
    CV_Assert( 0 <= coi && coi < dcn && scn == 1 );

    int ch[] = { 0, coi };

    Mat src = _src.getMat(), dst = _dst.getMat();
    mixChannels(&src, 1, &dst, 1, ch, 1);
}

/****************************************************************************************\
*                                convertScale[Abs]                                       *
\****************************************************************************************/

namespace cv
{

template<typename T, typename DT, typename WT>
struct cvtScaleAbs_SIMD
{
    int operator () (const T *, DT *, int, WT, WT) const
    {
        return 0;
    }
};

template<typename T, typename DT, typename WT> static void
cvtScaleAbs_( const T* src, size_t sstep,
              DT* dst, size_t dstep, Size size,
              WT scale, WT shift )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);
    cvtScaleAbs_SIMD<T, DT, WT> vop;

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = vop(src, dst, size.width, scale, shift);

        #if CV_ENABLE_UNROLLED
        for( ; x <= size.width - 4; x += 4 )
        {
            DT t0, t1;
            t0 = saturate_cast<DT>(std::abs(src[x]*scale + shift));
            t1 = saturate_cast<DT>(std::abs(src[x+1]*scale + shift));
            dst[x] = t0; dst[x+1] = t1;
            t0 = saturate_cast<DT>(std::abs(src[x+2]*scale + shift));
            t1 = saturate_cast<DT>(std::abs(src[x+3]*scale + shift));
            dst[x+2] = t0; dst[x+3] = t1;
        }
        #endif
        for( ; x < size.width; x++ )
            dst[x] = saturate_cast<DT>(std::abs(src[x]*scale + shift));
    }
}

template <typename T, typename DT, typename WT>
struct cvtScale_SIMD
{
    int operator () (const T *, DT *, int, WT, WT) const
    {
        return 0;
    }
};

template<typename T, typename DT, typename WT> static void
cvtScale_( const T* src, size_t sstep,
           DT* dst, size_t dstep, Size size,
           WT scale, WT shift )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    cvtScale_SIMD<T, DT, WT> vop;

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = vop(src, dst, size.width, scale, shift);

        #if CV_ENABLE_UNROLLED
        for( ; x <= size.width - 4; x += 4 )
        {
            DT t0, t1;
            t0 = saturate_cast<DT>(src[x]*scale + shift);
            t1 = saturate_cast<DT>(src[x+1]*scale + shift);
            dst[x] = t0; dst[x+1] = t1;
            t0 = saturate_cast<DT>(src[x+2]*scale + shift);
            t1 = saturate_cast<DT>(src[x+3]*scale + shift);
            dst[x+2] = t0; dst[x+3] = t1;
        }
        #endif

        for( ; x < size.width; x++ )
            dst[x] = saturate_cast<DT>(src[x]*scale + shift);
    }
}

//vz optimized template specialization
template<> void
cvtScale_<short, short, float>( const short* src, size_t sstep,
           short* dst, size_t dstep, Size size,
           float scale, float shift )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = 0;
        #if CV_NEON
        float32x4_t v_shift = vdupq_n_f32(shift);
        for(; x <= size.width - 8; x += 8 )
        {
            int16x8_t v_src = vld1q_s16(src + x);
            float32x4_t v_tmp1 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src)));
            float32x4_t v_tmp2 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src)));

            v_tmp1 = vaddq_f32(vmulq_n_f32(v_tmp1, scale), v_shift);
            v_tmp2 = vaddq_f32(vmulq_n_f32(v_tmp2, scale), v_shift);

            vst1q_s16(dst + x, vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_tmp1)),
                                            vqmovn_s32(cv_vrndq_s32_f32(v_tmp2))));
        }
        #endif

        for(; x < size.width; x++ )
            dst[x] = saturate_cast<short>(src[x]*scale + shift);
    }
}

template<> void
cvtScale_<short, int, float>( const short* src, size_t sstep,
           int* dst, size_t dstep, Size size,
           float scale, float shift )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = 0;

        for(; x < size.width; x++ )
            dst[x] = saturate_cast<int>(src[x]*scale + shift);
    }
}

template <typename T, typename DT>
struct Cvt_SIMD
{
    int operator() (const T *, DT *, int) const
    {
        return 0;
    }
};

#if !CV_FP16_TYPE
// const numbers for floating points format
const unsigned int kShiftSignificand    = 13;
const unsigned int kMaskFp16Significand = 0x3ff;
const unsigned int kBiasFp16Exponent    = 15;
const unsigned int kBiasFp32Exponent    = 127;
#endif

#if CV_FP16_TYPE
static float convertFp16SW(short fp16)
{
    // Fp16 -> Fp32
    Cv16suf a;
    a.i = fp16;
    return (float)a.h;
}
#else
static float convertFp16SW(short fp16)
{
    // Fp16 -> Fp32
    Cv16suf b;
    b.i = fp16;
    int exponent    = b.fmt.exponent - kBiasFp16Exponent;
    int significand = b.fmt.significand;

    Cv32suf a;
    a.i = 0;
    a.fmt.sign = b.fmt.sign; // sign bit
    if( exponent == 16 )
    {
        // Inf or NaN
        a.i = a.i | 0x7F800000;
        if( significand != 0 )
        {
            // NaN
#if defined(__x86_64__) || defined(_M_X64)
            // 64bit
            a.i = a.i | 0x7FC00000;
#endif
            a.fmt.significand = a.fmt.significand | (significand << kShiftSignificand);
        }
        return a.f;
    }
    else if ( exponent == -15 )
    {
        // subnormal in Fp16
        if( significand == 0 )
        {
            // zero
            return a.f;
        }
        else
        {
            int shift = -1;
            while( ( significand & 0x400 ) == 0 )
            {
                significand = significand << 1;
                shift++;
            }
            significand = significand & kMaskFp16Significand;
            exponent -= shift;
        }
    }

    a.fmt.exponent = (exponent+kBiasFp32Exponent);
    a.fmt.significand = significand << kShiftSignificand;
    return a.f;
}
#endif

#if CV_FP16_TYPE
static short convertFp16SW(float fp32)
{
    // Fp32 -> Fp16
    Cv16suf a;
    a.h = (__fp16)fp32;
    return a.i;
}
#else
static short convertFp16SW(float fp32)
{
    // Fp32 -> Fp16
    Cv32suf a;
    a.f = fp32;
    int exponent    = a.fmt.exponent - kBiasFp32Exponent;
    int significand = a.fmt.significand;

    Cv16suf result;
    result.i = 0;
    unsigned int absolute = a.i & 0x7fffffff;
    if( 0x477ff000 <= absolute )
    {
        // Inf in Fp16
        result.i = result.i | 0x7C00;
        if( exponent == 128 && significand != 0 )
        {
            // NaN
            result.i = (short)( result.i | 0x200 | ( significand >> kShiftSignificand ) );
        }
    }
    else if ( absolute < 0x33000001 )
    {
        // too small for fp16
        result.i = 0;
    }
    else if ( absolute < 0x33c00000 )
    {
        result.i = 1;
    }
    else if ( absolute < 0x34200001 )
    {
        result.i = 2;
    }
    else if ( absolute < 0x387fe000 )
    {
        // subnormal in Fp16
        int fp16Significand = significand | 0x800000;
        int bitShift = (-exponent) - 1;
        fp16Significand = fp16Significand >> bitShift;

        // special cases to round up
        bitShift = exponent + 24;
        int threshold = ( ( 0x400000 >> bitShift ) | ( ( ( significand & ( 0x800000 >> bitShift ) ) >> ( 126 - a.fmt.exponent ) ) ^ 1 ) );
        if( threshold <= ( significand & ( 0xffffff >> ( exponent + 25 ) ) ) )
        {
            fp16Significand++;
        }
        result.i = (short)fp16Significand;
    }
    else
    {
        // usual situation
        // exponent
        result.fmt.exponent = ( exponent + kBiasFp16Exponent );

        // significand;
        short fp16Significand = (short)(significand >> kShiftSignificand);
        result.fmt.significand = fp16Significand;

        // special cases to round up
        short lsb10bitsFp32 = (significand & 0x1fff);
        short threshold = 0x1000 + ( ( fp16Significand & 0x1 ) ? 0 : 1 );
        if( threshold <= lsb10bitsFp32 )
        {
            result.i++;
        }
        else if ( fp16Significand == 0x3ff && exponent == -15)
        {
            result.i++;
        }
    }

    // sign bit
    result.fmt.sign = a.fmt.sign;
    return result.i;
}
#endif

// template for FP16 HW conversion function
template<typename T, typename DT> static void
cvtScaleHalf_( const T* src, size_t sstep, DT* dst, size_t dstep, Size size);

template<> void
cvtScaleHalf_<float, short>( const float* src, size_t sstep, short* dst, size_t dstep, Size size)
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    if( checkHardwareSupport(CV_CPU_FP16) )
    {
        for( ; size.height--; src += sstep, dst += dstep )
        {
            int x = 0;

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86) || defined(i386)
            if ( ( (intptr_t)dst & 0xf ) == 0 )
#endif
            {
#if CV_FP16 && CV_SIMD128
                for ( ; x <= size.width - 4; x += 4)
                {
                    v_float32x4 v_src = v_load(src + x);

                    v_float16x4 v_dst = v_cvt_f16(v_src);

                    v_store_f16(dst + x, v_dst);
                }
#endif
            }
            for ( ; x < size.width; x++ )
            {
                dst[x] = convertFp16SW(src[x]);
            }
        }
    }
    else
    {
        for( ; size.height--; src += sstep, dst += dstep )
        {
            int x = 0;
            for ( ; x < size.width; x++ )
            {
                dst[x] = convertFp16SW(src[x]);
            }
        }
    }
}

template<> void
cvtScaleHalf_<short, float>( const short* src, size_t sstep, float* dst, size_t dstep, Size size)
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    if( checkHardwareSupport(CV_CPU_FP16) )
    {
        for( ; size.height--; src += sstep, dst += dstep )
        {
            int x = 0;

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86) || defined(i386)
            if ( ( (intptr_t)src & 0xf ) == 0 )
#endif
            {
#if CV_FP16 && CV_SIMD128
                for ( ; x <= size.width - 4; x += 4)
                {
                    v_float16x4 v_src = v_load_f16(src + x);

                    v_float32x4 v_dst = v_cvt_f32(v_src);

                    v_store(dst + x, v_dst);
                }
#endif
            }
            for ( ; x < size.width; x++ )
            {
                dst[x] = convertFp16SW(src[x]);
            }
        }
    }
    else
    {
        for( ; size.height--; src += sstep, dst += dstep )
        {
            int x = 0;
            for ( ; x < size.width; x++ )
            {
                dst[x] = convertFp16SW(src[x]);
            }
        }
    }
}

template<typename T, typename DT> static void
cvt_( const T* src, size_t sstep,
      DT* dst, size_t dstep, Size size )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);
    Cvt_SIMD<T, DT> vop;

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = vop(src, dst, size.width);
        #if CV_ENABLE_UNROLLED
        for( ; x <= size.width - 4; x += 4 )
        {
            DT t0, t1;
            t0 = saturate_cast<DT>(src[x]);
            t1 = saturate_cast<DT>(src[x+1]);
            dst[x] = t0; dst[x+1] = t1;
            t0 = saturate_cast<DT>(src[x+2]);
            t1 = saturate_cast<DT>(src[x+3]);
            dst[x+2] = t0; dst[x+3] = t1;
        }
        #endif
        for( ; x < size.width; x++ )
            dst[x] = saturate_cast<DT>(src[x]);
    }
}

//vz optimized template specialization, test Core_ConvertScale/ElemWiseTest
template<>  void
cvt_<float, short>( const float* src, size_t sstep,
     short* dst, size_t dstep, Size size )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = 0;
        for( ; x < size.width; x++ )
            dst[x] = saturate_cast<short>(src[x]);
    }

}


template<typename T> static void
cpy_( const T* src, size_t sstep, T* dst, size_t dstep, Size size )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
        memcpy(dst, src, size.width*sizeof(src[0]));
}

#define DEF_CVT_SCALE_ABS_FUNC(suffix, tfunc, stype, dtype, wtype) \
static void cvtScaleAbs##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double* scale) \
{ \
    tfunc(src, sstep, dst, dstep, size, (wtype)scale[0], (wtype)scale[1]); \
}

#define DEF_CVT_SCALE_FP16_FUNC(suffix, stype, dtype) \
static void cvtScaleHalf##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
dtype* dst, size_t dstep, Size size, double*) \
{ \
    cvtScaleHalf_<stype,dtype>(src, sstep, dst, dstep, size); \
}

#define DEF_CVT_SCALE_FUNC(suffix, stype, dtype, wtype) \
static void cvtScale##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
dtype* dst, size_t dstep, Size size, double* scale) \
{ \
    cvtScale_(src, sstep, dst, dstep, size, (wtype)scale[0], (wtype)scale[1]); \
}

#define DEF_CVT_FUNC_F(suffix, stype, dtype, ippFavor) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double*) \
{ \
    cvt_(src, sstep, dst, dstep, size); \
}
#define DEF_CVT_FUNC_F2 DEF_CVT_FUNC_F

#define DEF_CVT_FUNC(suffix, stype, dtype) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double*) \
{ \
    cvt_(src, sstep, dst, dstep, size); \
}

#define DEF_CPY_FUNC(suffix, stype) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         stype* dst, size_t dstep, Size size, double*) \
{ \
    cpy_(src, sstep, dst, dstep, size); \
}


DEF_CVT_SCALE_ABS_FUNC(8u, cvtScaleAbs_, uchar, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(8s8u, cvtScaleAbs_, schar, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(16u8u, cvtScaleAbs_, ushort, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(16s8u, cvtScaleAbs_, short, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(32s8u, cvtScaleAbs_, int, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(32f8u, cvtScaleAbs_, float, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(64f8u, cvtScaleAbs_, double, uchar, float)

DEF_CVT_SCALE_FP16_FUNC(32f16f, float, short)
DEF_CVT_SCALE_FP16_FUNC(16f32f, short, float)

DEF_CVT_SCALE_FUNC(8u,     uchar, uchar, float)
DEF_CVT_SCALE_FUNC(8s8u,   schar, uchar, float)
DEF_CVT_SCALE_FUNC(16u8u,  ushort, uchar, float)
DEF_CVT_SCALE_FUNC(16s8u,  short, uchar, float)
DEF_CVT_SCALE_FUNC(32s8u,  int, uchar, float)
DEF_CVT_SCALE_FUNC(32f8u,  float, uchar, float)
DEF_CVT_SCALE_FUNC(64f8u,  double, uchar, float)

DEF_CVT_SCALE_FUNC(8u8s,   uchar, schar, float)
DEF_CVT_SCALE_FUNC(8s,     schar, schar, float)
DEF_CVT_SCALE_FUNC(16u8s,  ushort, schar, float)
DEF_CVT_SCALE_FUNC(16s8s,  short, schar, float)
DEF_CVT_SCALE_FUNC(32s8s,  int, schar, float)
DEF_CVT_SCALE_FUNC(32f8s,  float, schar, float)
DEF_CVT_SCALE_FUNC(64f8s,  double, schar, float)

DEF_CVT_SCALE_FUNC(8u16u,  uchar, ushort, float)
DEF_CVT_SCALE_FUNC(8s16u,  schar, ushort, float)
DEF_CVT_SCALE_FUNC(16u,    ushort, ushort, float)
DEF_CVT_SCALE_FUNC(16s16u, short, ushort, float)
DEF_CVT_SCALE_FUNC(32s16u, int, ushort, float)
DEF_CVT_SCALE_FUNC(32f16u, float, ushort, float)
DEF_CVT_SCALE_FUNC(64f16u, double, ushort, float)

DEF_CVT_SCALE_FUNC(8u16s,  uchar, short, float)
DEF_CVT_SCALE_FUNC(8s16s,  schar, short, float)
DEF_CVT_SCALE_FUNC(16u16s, ushort, short, float)
DEF_CVT_SCALE_FUNC(16s,    short, short, float)
DEF_CVT_SCALE_FUNC(32s16s, int, short, float)
DEF_CVT_SCALE_FUNC(32f16s, float, short, float)
DEF_CVT_SCALE_FUNC(64f16s, double, short, float)

DEF_CVT_SCALE_FUNC(8u32s,  uchar, int, float)
DEF_CVT_SCALE_FUNC(8s32s,  schar, int, float)
DEF_CVT_SCALE_FUNC(16u32s, ushort, int, float)
DEF_CVT_SCALE_FUNC(16s32s, short, int, float)
DEF_CVT_SCALE_FUNC(32s,    int, int, double)
DEF_CVT_SCALE_FUNC(32f32s, float, int, float)
DEF_CVT_SCALE_FUNC(64f32s, double, int, double)

DEF_CVT_SCALE_FUNC(8u32f,  uchar, float, float)
DEF_CVT_SCALE_FUNC(8s32f,  schar, float, float)
DEF_CVT_SCALE_FUNC(16u32f, ushort, float, float)
DEF_CVT_SCALE_FUNC(16s32f, short, float, float)
DEF_CVT_SCALE_FUNC(32s32f, int, float, double)
DEF_CVT_SCALE_FUNC(32f,    float, float, float)
DEF_CVT_SCALE_FUNC(64f32f, double, float, double)

DEF_CVT_SCALE_FUNC(8u64f,  uchar, double, double)
DEF_CVT_SCALE_FUNC(8s64f,  schar, double, double)
DEF_CVT_SCALE_FUNC(16u64f, ushort, double, double)
DEF_CVT_SCALE_FUNC(16s64f, short, double, double)
DEF_CVT_SCALE_FUNC(32s64f, int, double, double)
DEF_CVT_SCALE_FUNC(32f64f, float, double, double)
DEF_CVT_SCALE_FUNC(64f,    double, double, double)

DEF_CPY_FUNC(8u,     uchar)
DEF_CVT_FUNC_F(8s8u,   schar, uchar, 8s8u_C1Rs)
DEF_CVT_FUNC_F(16u8u,  ushort, uchar, 16u8u_C1R)
DEF_CVT_FUNC_F(16s8u,  short, uchar, 16s8u_C1R)
DEF_CVT_FUNC_F(32s8u,  int, uchar, 32s8u_C1R)
DEF_CVT_FUNC_F2(32f8u,  float, uchar, 32f8u_C1RSfs)
DEF_CVT_FUNC(64f8u,  double, uchar)

DEF_CVT_FUNC_F2(8u8s,   uchar, schar, 8u8s_C1RSfs)
DEF_CVT_FUNC_F2(16u8s,  ushort, schar, 16u8s_C1RSfs)
DEF_CVT_FUNC_F2(16s8s,  short, schar, 16s8s_C1RSfs)
DEF_CVT_FUNC_F(32s8s,  int, schar, 32s8s_C1R)
DEF_CVT_FUNC_F2(32f8s,  float, schar, 32f8s_C1RSfs)
DEF_CVT_FUNC(64f8s,  double, schar)

DEF_CVT_FUNC_F(8u16u,  uchar, ushort, 8u16u_C1R)
DEF_CVT_FUNC_F(8s16u,  schar, ushort, 8s16u_C1Rs)
DEF_CPY_FUNC(16u,    ushort)
DEF_CVT_FUNC_F(16s16u, short, ushort, 16s16u_C1Rs)
DEF_CVT_FUNC_F2(32s16u, int, ushort, 32s16u_C1RSfs)
DEF_CVT_FUNC_F2(32f16u, float, ushort, 32f16u_C1RSfs)
DEF_CVT_FUNC(64f16u, double, ushort)

DEF_CVT_FUNC_F(8u16s,  uchar, short, 8u16s_C1R)
DEF_CVT_FUNC_F(8s16s,  schar, short, 8s16s_C1R)
DEF_CVT_FUNC_F2(16u16s, ushort, short, 16u16s_C1RSfs)
DEF_CVT_FUNC_F2(32s16s, int, short, 32s16s_C1RSfs)
DEF_CVT_FUNC(32f16s, float, short)
DEF_CVT_FUNC(64f16s, double, short)

DEF_CVT_FUNC_F(8u32s,  uchar, int, 8u32s_C1R)
DEF_CVT_FUNC_F(8s32s,  schar, int, 8s32s_C1R)
DEF_CVT_FUNC_F(16u32s, ushort, int, 16u32s_C1R)
DEF_CVT_FUNC_F(16s32s, short, int, 16s32s_C1R)
DEF_CPY_FUNC(32s,    int)
DEF_CVT_FUNC_F2(32f32s, float, int, 32f32s_C1RSfs)
DEF_CVT_FUNC(64f32s, double, int)

DEF_CVT_FUNC_F(8u32f,  uchar, float, 8u32f_C1R)
DEF_CVT_FUNC_F(8s32f,  schar, float, 8s32f_C1R)
DEF_CVT_FUNC_F(16u32f, ushort, float, 16u32f_C1R)
DEF_CVT_FUNC_F(16s32f, short, float, 16s32f_C1R)
DEF_CVT_FUNC_F(32s32f, int, float, 32s32f_C1R)
DEF_CVT_FUNC(64f32f, double, float)

DEF_CVT_FUNC(8u64f,  uchar, double)
DEF_CVT_FUNC(8s64f,  schar, double)
DEF_CVT_FUNC(16u64f, ushort, double)
DEF_CVT_FUNC(16s64f, short, double)
DEF_CVT_FUNC(32s64f, int, double)
DEF_CVT_FUNC(32f64f, float, double)
DEF_CPY_FUNC(64s,    int64)

static BinaryFunc getCvtScaleAbsFunc(int depth)
{
    static BinaryFunc cvtScaleAbsTab[] =
    {
        (BinaryFunc)cvtScaleAbs8u, (BinaryFunc)cvtScaleAbs8s8u, (BinaryFunc)cvtScaleAbs16u8u,
        (BinaryFunc)cvtScaleAbs16s8u, (BinaryFunc)cvtScaleAbs32s8u, (BinaryFunc)cvtScaleAbs32f8u,
        (BinaryFunc)cvtScaleAbs64f8u, 0
    };

    return cvtScaleAbsTab[depth];
}

BinaryFunc getConvertFuncFp16(int ddepth)
{
    static BinaryFunc cvtTab[] =
    {
        0, 0, 0,
        (BinaryFunc)(cvtScaleHalf32f16f), 0, (BinaryFunc)(cvtScaleHalf16f32f),
        0, 0,
    };
    return cvtTab[CV_MAT_DEPTH(ddepth)];
}

BinaryFunc getConvertFunc(int sdepth, int ddepth)
{
    static BinaryFunc cvtTab[][8] =
    {
        {
            (BinaryFunc)(cvt8u), (BinaryFunc)GET_OPTIMIZED(cvt8s8u), (BinaryFunc)GET_OPTIMIZED(cvt16u8u),
            (BinaryFunc)GET_OPTIMIZED(cvt16s8u), (BinaryFunc)GET_OPTIMIZED(cvt32s8u), (BinaryFunc)GET_OPTIMIZED(cvt32f8u),
            (BinaryFunc)GET_OPTIMIZED(cvt64f8u), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u8s), (BinaryFunc)cvt8u, (BinaryFunc)GET_OPTIMIZED(cvt16u8s),
            (BinaryFunc)GET_OPTIMIZED(cvt16s8s), (BinaryFunc)GET_OPTIMIZED(cvt32s8s), (BinaryFunc)GET_OPTIMIZED(cvt32f8s),
            (BinaryFunc)GET_OPTIMIZED(cvt64f8s), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u16u), (BinaryFunc)GET_OPTIMIZED(cvt8s16u), (BinaryFunc)cvt16u,
            (BinaryFunc)GET_OPTIMIZED(cvt16s16u), (BinaryFunc)GET_OPTIMIZED(cvt32s16u), (BinaryFunc)GET_OPTIMIZED(cvt32f16u),
            (BinaryFunc)GET_OPTIMIZED(cvt64f16u), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u16s), (BinaryFunc)GET_OPTIMIZED(cvt8s16s), (BinaryFunc)GET_OPTIMIZED(cvt16u16s),
            (BinaryFunc)cvt16u, (BinaryFunc)GET_OPTIMIZED(cvt32s16s), (BinaryFunc)GET_OPTIMIZED(cvt32f16s),
            (BinaryFunc)GET_OPTIMIZED(cvt64f16s), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u32s), (BinaryFunc)GET_OPTIMIZED(cvt8s32s), (BinaryFunc)GET_OPTIMIZED(cvt16u32s),
            (BinaryFunc)GET_OPTIMIZED(cvt16s32s), (BinaryFunc)cvt32s, (BinaryFunc)GET_OPTIMIZED(cvt32f32s),
            (BinaryFunc)GET_OPTIMIZED(cvt64f32s), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u32f), (BinaryFunc)GET_OPTIMIZED(cvt8s32f), (BinaryFunc)GET_OPTIMIZED(cvt16u32f),
            (BinaryFunc)GET_OPTIMIZED(cvt16s32f), (BinaryFunc)GET_OPTIMIZED(cvt32s32f), (BinaryFunc)cvt32s,
            (BinaryFunc)GET_OPTIMIZED(cvt64f32f), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u64f), (BinaryFunc)GET_OPTIMIZED(cvt8s64f), (BinaryFunc)GET_OPTIMIZED(cvt16u64f),
            (BinaryFunc)GET_OPTIMIZED(cvt16s64f), (BinaryFunc)GET_OPTIMIZED(cvt32s64f), (BinaryFunc)GET_OPTIMIZED(cvt32f64f),
            (BinaryFunc)(cvt64s), 0
        },
        {
            0, 0, 0, 0, 0, 0, 0, 0
        }
    };

    return cvtTab[CV_MAT_DEPTH(ddepth)][CV_MAT_DEPTH(sdepth)];
}

static BinaryFunc getConvertScaleFunc(int sdepth, int ddepth)
{
    static BinaryFunc cvtScaleTab[][8] =
    {
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u), (BinaryFunc)GET_OPTIMIZED(cvtScale8s8u), (BinaryFunc)GET_OPTIMIZED(cvtScale16u8u),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s8u), (BinaryFunc)GET_OPTIMIZED(cvtScale32s8u), (BinaryFunc)GET_OPTIMIZED(cvtScale32f8u),
            (BinaryFunc)cvtScale64f8u, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u8s), (BinaryFunc)GET_OPTIMIZED(cvtScale8s), (BinaryFunc)GET_OPTIMIZED(cvtScale16u8s),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s8s), (BinaryFunc)GET_OPTIMIZED(cvtScale32s8s), (BinaryFunc)GET_OPTIMIZED(cvtScale32f8s),
            (BinaryFunc)cvtScale64f8s, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u16u), (BinaryFunc)GET_OPTIMIZED(cvtScale8s16u), (BinaryFunc)GET_OPTIMIZED(cvtScale16u),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s16u), (BinaryFunc)GET_OPTIMIZED(cvtScale32s16u), (BinaryFunc)GET_OPTIMIZED(cvtScale32f16u),
            (BinaryFunc)cvtScale64f16u, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u16s), (BinaryFunc)GET_OPTIMIZED(cvtScale8s16s), (BinaryFunc)GET_OPTIMIZED(cvtScale16u16s),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s), (BinaryFunc)GET_OPTIMIZED(cvtScale32s16s), (BinaryFunc)GET_OPTIMIZED(cvtScale32f16s),
            (BinaryFunc)cvtScale64f16s, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u32s), (BinaryFunc)GET_OPTIMIZED(cvtScale8s32s), (BinaryFunc)GET_OPTIMIZED(cvtScale16u32s),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s32s), (BinaryFunc)GET_OPTIMIZED(cvtScale32s), (BinaryFunc)GET_OPTIMIZED(cvtScale32f32s),
            (BinaryFunc)cvtScale64f32s, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u32f), (BinaryFunc)GET_OPTIMIZED(cvtScale8s32f), (BinaryFunc)GET_OPTIMIZED(cvtScale16u32f),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s32f), (BinaryFunc)GET_OPTIMIZED(cvtScale32s32f), (BinaryFunc)GET_OPTIMIZED(cvtScale32f),
            (BinaryFunc)cvtScale64f32f, 0
        },
        {
            (BinaryFunc)cvtScale8u64f, (BinaryFunc)cvtScale8s64f, (BinaryFunc)cvtScale16u64f,
            (BinaryFunc)cvtScale16s64f, (BinaryFunc)cvtScale32s64f, (BinaryFunc)cvtScale32f64f,
            (BinaryFunc)cvtScale64f, 0
        },
        {
            0, 0, 0, 0, 0, 0, 0, 0
        }
    };

    return cvtScaleTab[CV_MAT_DEPTH(ddepth)][CV_MAT_DEPTH(sdepth)];
}

}

void cv::convertScaleAbs( InputArray _src, OutputArray _dst, double alpha, double beta )
{
    CV_INSTRUMENT_REGION()

    Mat src = _src.getMat();
    int cn = src.channels();
    double scale[] = {alpha, beta};
    _dst.create( src.dims, src.size, CV_8UC(cn) );
    Mat dst = _dst.getMat();
    BinaryFunc func = getCvtScaleAbsFunc(src.depth());
    CV_Assert( func != 0 );

    if( src.dims <= 2 )
    {
        Size sz = getContinuousSize(src, dst, cn);
        func( src.ptr(), src.step, 0, 0, dst.ptr(), dst.step, sz, scale );
    }
    else
    {
        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs);
        Size sz((int)it.size*cn, 1);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func( ptrs[0], 0, 0, 0, ptrs[1], 0, sz, scale );
    }
}

void cv::convertFp16( InputArray _src, OutputArray _dst)
{
    CV_INSTRUMENT_REGION()

    Mat src = _src.getMat();
    int ddepth = 0;

    switch( src.depth() )
    {
    case CV_32F:
        ddepth = CV_16S;
        break;
    case CV_16S:
        ddepth = CV_32F;
        break;
    default:
        CV_Error(Error::StsUnsupportedFormat, "Unsupported input depth");
        return;
    }

    int type = CV_MAKETYPE(ddepth, src.channels());
    _dst.create( src.dims, src.size, type );
    Mat dst = _dst.getMat();
    BinaryFunc func = getConvertFuncFp16(ddepth);
    int cn = src.channels();
    CV_Assert( func != 0 );

    if( src.dims <= 2 )
    {
        Size sz = getContinuousSize(src, dst, cn);
        func( src.data, src.step, 0, 0, dst.data, dst.step, sz, 0);
    }
    else
    {
        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs);
        Size sz((int)(it.size*cn), 1);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func(ptrs[0], 1, 0, 0, ptrs[1], 1, sz, 0);
    }
}

void cv::Mat::convertTo(OutputArray _dst, int _type, double alpha, double beta) const
{
    CV_INSTRUMENT_REGION()

    bool noScale = fabs(alpha-1) < DBL_EPSILON && fabs(beta) < DBL_EPSILON;

    if( _type < 0 )
        _type = _dst.fixedType() ? _dst.type() : type();
    else
        _type = CV_MAKETYPE(CV_MAT_DEPTH(_type), channels());

    int sdepth = depth(), ddepth = CV_MAT_DEPTH(_type);
    if( sdepth == ddepth && noScale )
    {
        copyTo(_dst);
        return;
    }

    Mat src = *this;

    BinaryFunc func = noScale ? getConvertFunc(sdepth, ddepth) : getConvertScaleFunc(sdepth, ddepth);
    double scale[] = {alpha, beta};
    int cn = channels();
    CV_Assert( func != 0 );

    if( dims <= 2 )
    {
        _dst.create( size(), _type );
        Mat dst = _dst.getMat();
        Size sz = getContinuousSize(src, dst, cn);
        func( src.data, src.step, 0, 0, dst.data, dst.step, sz, scale );
    }
    else
    {
        _dst.create( dims, size, _type );
        Mat dst = _dst.getMat();
        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs);
        Size sz((int)(it.size*cn), 1);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func(ptrs[0], 1, 0, 0, ptrs[1], 1, sz, scale);
    }
}

void cv::normalize( InputArray _src, InputOutputArray _dst, double a, double b,
                    int norm_type, int rtype, InputArray _mask )
{
    CV_INSTRUMENT_REGION()

    double scale = 1, shift = 0;
    if( norm_type == CV_MINMAX )
    {
        double smin = 0, smax = 0;
        double dmin = MIN( a, b ), dmax = MAX( a, b );
        minMaxIdx( _src, &smin, &smax, 0, 0, _mask );
        scale = (dmax - dmin)*(smax - smin > DBL_EPSILON ? 1./(smax - smin) : 0);
        shift = dmin - smin*scale;
    }
    else if( norm_type == CV_L2 || norm_type == CV_L1 || norm_type == CV_C )
    {
        scale = norm( _src, norm_type, _mask );
        scale = scale > DBL_EPSILON ? a/scale : 0.;
        shift = 0;
    }
    else
        CV_Error( CV_StsBadArg, "Unknown/unsupported norm type" );

    int type = _src.type(), depth = CV_MAT_DEPTH(type);
    if( rtype < 0 )
        rtype = _dst.fixedType() ? _dst.depth() : depth;

    Mat src = _src.getMat();
    if( _mask.empty() )
        src.convertTo( _dst, rtype, scale, shift );
    else
    {
        Mat temp;
        src.convertTo( temp, rtype, scale, shift );
        temp.copyTo( _dst, _mask );
    }
}

CV_IMPL void
cvSplit( const void* srcarr, void* dstarr0, void* dstarr1, void* dstarr2, void* dstarr3 )
{
    void* dptrs[] = { dstarr0, dstarr1, dstarr2, dstarr3 };
    cv::Mat src = cv::cvarrToMat(srcarr);
    int i, j, nz = 0;
    for( i = 0; i < 4; i++ )
        nz += dptrs[i] != 0;
    CV_Assert( nz > 0 );
    std::vector<cv::Mat> dvec(nz);
    std::vector<int> pairs(nz*2);

    for( i = j = 0; i < 4; i++ )
    {
        if( dptrs[i] != 0 )
        {
            dvec[j] = cv::cvarrToMat(dptrs[i]);
            CV_Assert( dvec[j].size() == src.size() );
            CV_Assert( dvec[j].depth() == src.depth() );
            CV_Assert( dvec[j].channels() == 1 );
            CV_Assert( i < src.channels() );
            pairs[j*2] = i;
            pairs[j*2+1] = j;
            j++;
        }
    }
    if( nz == src.channels() )
        cv::split( src, dvec );
    else
    {
        cv::mixChannels( &src, 1, &dvec[0], nz, &pairs[0], nz );
    }
}


CV_IMPL void
cvMerge( const void* srcarr0, const void* srcarr1, const void* srcarr2,
         const void* srcarr3, void* dstarr )
{
    const void* sptrs[] = { srcarr0, srcarr1, srcarr2, srcarr3 };
    cv::Mat dst = cv::cvarrToMat(dstarr);
    int i, j, nz = 0;
    for( i = 0; i < 4; i++ )
        nz += sptrs[i] != 0;
    CV_Assert( nz > 0 );
    std::vector<cv::Mat> svec(nz);
    std::vector<int> pairs(nz*2);

    for( i = j = 0; i < 4; i++ )
    {
        if( sptrs[i] != 0 )
        {
            svec[j] = cv::cvarrToMat(sptrs[i]);
            CV_Assert( svec[j].size == dst.size &&
                svec[j].depth() == dst.depth() &&
                svec[j].channels() == 1 && i < dst.channels() );
            pairs[j*2] = j;
            pairs[j*2+1] = i;
            j++;
        }
    }

    if( nz == dst.channels() )
        cv::merge( svec, dst );
    else
    {
        cv::mixChannels( &svec[0], nz, &dst, 1, &pairs[0], nz );
    }
}


CV_IMPL void
cvMixChannels( const CvArr** src, int src_count,
               CvArr** dst, int dst_count,
               const int* from_to, int pair_count )
{
    cv::AutoBuffer<cv::Mat> buf(src_count + dst_count);

    int i;
    for( i = 0; i < src_count; i++ )
        buf[i] = cv::cvarrToMat(src[i]);
    for( i = 0; i < dst_count; i++ )
        buf[i+src_count] = cv::cvarrToMat(dst[i]);
    cv::mixChannels(&buf[0], src_count, &buf[src_count], dst_count, from_to, pair_count);
}

CV_IMPL void
cvConvertScaleAbs( const void* srcarr, void* dstarr,
                   double scale, double shift )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src.size == dst.size && dst.type() == CV_8UC(src.channels()));
    cv::convertScaleAbs( src, dst, scale, shift );
}

CV_IMPL void
cvConvertScale( const void* srcarr, void* dstarr,
                double scale, double shift )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.size == dst.size && src.channels() == dst.channels() );
    src.convertTo(dst, dst.type(), scale, shift);
}

CV_IMPL void cvNormalize( const CvArr* srcarr, CvArr* dstarr,
                          double a, double b, int norm_type, const CvArr* maskarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), mask;
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    CV_Assert( dst.size() == src.size() && src.channels() == dst.channels() );
    cv::normalize( src, dst, a, b, norm_type, dst.type(), mask );
}

/* End of file. */
