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
// Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014, Itseez Inc., all rights reserved.
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

#include <limits>

#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))

namespace cv
{


//////////////////////////// Bayer Pattern -> RGB conversion /////////////////////////////

template<typename T>
class SIMDBayerStubInterpolator_
{
public:
    int bayer2Gray(const T*, int, T*, int, int, int, int) const
    {
        return 0;
    }

    int bayer2RGB(const T*, int, T*, int, int) const
    {
        return 0;
    }

    int bayer2RGBA(const T*, int, T*, int, int) const
    {
        return 0;
    }

    int bayer2RGB_EA(const T*, int, T*, int, int) const
    {
        return 0;
    }
};

typedef SIMDBayerStubInterpolator_<uchar> SIMDBayerInterpolator_8u;

template<typename T, class SIMDInterpolator>
class Bayer2Gray_Invoker :
    public ParallelLoopBody
{
public:
    Bayer2Gray_Invoker(const Mat& _srcmat, Mat& _dstmat, int _start_with_green, bool _brow,
        const Size& _size, int _bcoeff, int _rcoeff) :
        ParallelLoopBody(), srcmat(_srcmat), dstmat(_dstmat), Start_with_green(_start_with_green),
        Brow(_brow), size(_size), Bcoeff(_bcoeff), Rcoeff(_rcoeff)
    {
    }

    virtual void operator ()(const Range& range) const
    {
        SIMDInterpolator vecOp;
        const int G2Y = 9617;
        const int SHIFT = 14;

        const T* bayer0 = srcmat.ptr<T>();
        int bayer_step = (int)(srcmat.step/sizeof(T));
        T* dst0 = (T*)dstmat.data;
        int dst_step = (int)(dstmat.step/sizeof(T));
        int bcoeff = Bcoeff, rcoeff = Rcoeff;
        int start_with_green = Start_with_green;
        bool brow = Brow;

        dst0 += dst_step + 1;

        if (range.start % 2)
        {
            brow = !brow;
            std::swap(bcoeff, rcoeff);
            start_with_green = !start_with_green;
        }

        bayer0 += range.start * bayer_step;
        dst0 += range.start * dst_step;

        for(int i = range.start ; i < range.end; ++i, bayer0 += bayer_step, dst0 += dst_step )
        {
            unsigned t0, t1, t2;
            const T* bayer = bayer0;
            T* dst = dst0;
            const T* bayer_end = bayer + size.width;

            if( size.width <= 0 )
            {
                dst[-1] = dst[size.width] = 0;
                continue;
            }

            if( start_with_green )
            {
                t0 = (bayer[1] + bayer[bayer_step*2+1])*rcoeff;
                t1 = (bayer[bayer_step] + bayer[bayer_step+2])*bcoeff;
                t2 = bayer[bayer_step+1]*(2*G2Y);

                dst[0] = (T)CV_DESCALE(t0 + t1 + t2, SHIFT+1);
                bayer++;
                dst++;
            }

            int delta = vecOp.bayer2Gray(bayer, bayer_step, dst, size.width, bcoeff, G2Y, rcoeff);
            bayer += delta;
            dst += delta;

            for( ; bayer <= bayer_end - 2; bayer += 2, dst += 2 )
            {
                t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] + bayer[bayer_step*2+2])*rcoeff;
                t1 = (bayer[1] + bayer[bayer_step] + bayer[bayer_step+2] + bayer[bayer_step*2+1])*G2Y;
                t2 = bayer[bayer_step+1]*(4*bcoeff);
                dst[0] = (T)CV_DESCALE(t0 + t1 + t2, SHIFT+2);

                t0 = (bayer[2] + bayer[bayer_step*2+2])*rcoeff;
                t1 = (bayer[bayer_step+1] + bayer[bayer_step+3])*bcoeff;
                t2 = bayer[bayer_step+2]*(2*G2Y);
                dst[1] = (T)CV_DESCALE(t0 + t1 + t2, SHIFT+1);
            }

            if( bayer < bayer_end )
            {
                t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] + bayer[bayer_step*2+2])*rcoeff;
                t1 = (bayer[1] + bayer[bayer_step] + bayer[bayer_step+2] + bayer[bayer_step*2+1])*G2Y;
                t2 = bayer[bayer_step+1]*(4*bcoeff);
                dst[0] = (T)CV_DESCALE(t0 + t1 + t2, SHIFT+2);
                bayer++;
                dst++;
            }

            dst0[-1] = dst0[0];
            dst0[size.width] = dst0[size.width-1];

            brow = !brow;
            std::swap(bcoeff, rcoeff);
            start_with_green = !start_with_green;
        }
    }

private:
    Mat srcmat;
    Mat dstmat;
    int Start_with_green;
    bool Brow;
    Size size;
    int Bcoeff, Rcoeff;
};

template<typename T, typename SIMDInterpolator>
static void Bayer2Gray_( const Mat& srcmat, Mat& dstmat, int code )
{
    const int R2Y = 4899;
    const int B2Y = 1868;

    Size size = srcmat.size();
    int bcoeff = B2Y, rcoeff = R2Y;
    int start_with_green = code == CV_BayerGB2GRAY || code == CV_BayerGR2GRAY;
    bool brow = true;

    if( code != CV_BayerBG2GRAY && code != CV_BayerGB2GRAY )
    {
        brow = false;
        std::swap(bcoeff, rcoeff);
    }
    size.height -= 2;
    size.width -= 2;

    if (size.height > 0)
    {
        Range range(0, size.height);
        Bayer2Gray_Invoker<T, SIMDInterpolator> invoker(srcmat, dstmat,
            start_with_green, brow, size, bcoeff, rcoeff);
        parallel_for_(range, invoker, dstmat.total()/static_cast<double>(1<<16));
    }

    size = dstmat.size();
    T* dst0 = dstmat.ptr<T>();
    int dst_step = (int)(dstmat.step/sizeof(T));
    if( size.height > 2 )
        for( int i = 0; i < size.width; i++ )
        {
            dst0[i] = dst0[i + dst_step];
            dst0[i + (size.height-1)*dst_step] = dst0[i + (size.height-2)*dst_step];
        }
    else
        for( int i = 0; i < size.width; i++ )
            dst0[i] = dst0[i + (size.height-1)*dst_step] = 0;
}

template <typename T>
struct Alpha
{
    static T value() { return std::numeric_limits<T>::max(); }
};

template <>
struct Alpha<float>
{
    static float value() { return 1.0f; }
};

template <typename T, typename SIMDInterpolator>
class Bayer2RGB_Invoker :
    public ParallelLoopBody
{
public:
    Bayer2RGB_Invoker(const Mat& _srcmat, Mat& _dstmat, int _start_with_green, int _blue, const Size& _size) :
        ParallelLoopBody(),
        srcmat(_srcmat), dstmat(_dstmat), Start_with_green(_start_with_green), Blue(_blue), size(_size)
    {
    }

    virtual void operator() (const Range& range) const
    {
        SIMDInterpolator vecOp;
        T alpha = Alpha<T>::value();
        int dcn = dstmat.channels();
        int dcn2 = dcn << 1;

        int bayer_step = (int)(srcmat.step/sizeof(T));
        const T* bayer0 = srcmat.ptr<T>() + bayer_step * range.start;

        int dst_step = (int)(dstmat.step/sizeof(T));
        T* dst0 = reinterpret_cast<T*>(dstmat.data) + (range.start + 1) * dst_step + dcn + 1;

        int blue = Blue, start_with_green = Start_with_green;
        if (range.start % 2)
        {
            blue = -blue;
            start_with_green = !start_with_green;
        }

        for (int i = range.start; i < range.end; bayer0 += bayer_step, dst0 += dst_step, ++i )
        {
            int t0, t1;
            const T* bayer = bayer0;
            T* dst = dst0;
            const T* bayer_end = bayer + size.width;

            // in case of when size.width <= 2
            if( size.width <= 0 )
            {
                if (dcn == 3)
                {
                    dst[-4] = dst[-3] = dst[-2] = dst[size.width*dcn-1] =
                    dst[size.width*dcn] = dst[size.width*dcn+1] = 0;
                }
                else
                {
                    dst[-5] = dst[-4] = dst[-3] = dst[size.width*dcn-1] =
                    dst[size.width*dcn] = dst[size.width*dcn+1] = 0;
                    dst[-2] = dst[size.width*dcn+2] = alpha;
                }
                continue;
            }

            if( start_with_green )
            {
                t0 = (bayer[1] + bayer[bayer_step*2+1] + 1) >> 1;
                t1 = (bayer[bayer_step] + bayer[bayer_step+2] + 1) >> 1;

                dst[-blue] = (T)t0;
                dst[0] = bayer[bayer_step+1];
                dst[blue] = (T)t1;
                if (dcn == 4)
                    dst[2] = alpha; // alpha channel

                bayer++;
                dst += dcn;
            }

            // simd optimization only for dcn == 3
            int delta = dcn == 4 ?
                vecOp.bayer2RGBA(bayer, bayer_step, dst, size.width, blue) :
                vecOp.bayer2RGB(bayer, bayer_step, dst, size.width, blue);
            bayer += delta;
            dst += delta*dcn;

            if (dcn == 3) // Bayer to BGR
            {
                if( blue > 0 )
                {
                    for( ; bayer <= bayer_end - 2; bayer += 2, dst += dcn2 )
                    {
                        t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] +
                              bayer[bayer_step*2+2] + 2) >> 2;
                        t1 = (bayer[1] + bayer[bayer_step] +
                              bayer[bayer_step+2] + bayer[bayer_step*2+1]+2) >> 2;
                        dst[-1] = (T)t0;
                        dst[0] = (T)t1;
                        dst[1] = bayer[bayer_step+1];

                        t0 = (bayer[2] + bayer[bayer_step*2+2] + 1) >> 1;
                        t1 = (bayer[bayer_step+1] + bayer[bayer_step+3] + 1) >> 1;
                        dst[2] = (T)t0;
                        dst[3] = bayer[bayer_step+2];
                        dst[4] = (T)t1;
                    }
                }
                else
                {
                    for( ; bayer <= bayer_end - 2; bayer += 2, dst += dcn2 )
                    {
                        t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] +
                              bayer[bayer_step*2+2] + 2) >> 2;
                        t1 = (bayer[1] + bayer[bayer_step] +
                              bayer[bayer_step+2] + bayer[bayer_step*2+1]+2) >> 2;
                        dst[1] = (T)t0;
                        dst[0] = (T)t1;
                        dst[-1] = bayer[bayer_step+1];

                        t0 = (bayer[2] + bayer[bayer_step*2+2] + 1) >> 1;
                        t1 = (bayer[bayer_step+1] + bayer[bayer_step+3] + 1) >> 1;
                        dst[4] = (T)t0;
                        dst[3] = bayer[bayer_step+2];
                        dst[2] = (T)t1;
                    }
                }
            }
            else // Bayer to BGRA
            {
                // if current row does not contain Blue pixels
                if( blue > 0 )
                {
                    for( ; bayer <= bayer_end - 2; bayer += 2, dst += dcn2 )
                    {
                        t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] +
                              bayer[bayer_step*2+2] + 2) >> 2;
                        t1 = (bayer[1] + bayer[bayer_step] +
                              bayer[bayer_step+2] + bayer[bayer_step*2+1]+2) >> 2;
                        dst[-1] = (T)t0;
                        dst[0] = (T)t1;
                        dst[1] = bayer[bayer_step+1];
                        dst[2] = alpha; // alpha channel

                        t0 = (bayer[2] + bayer[bayer_step*2+2] + 1) >> 1;
                        t1 = (bayer[bayer_step+1] + bayer[bayer_step+3] + 1) >> 1;
                        dst[3] = (T)t0;
                        dst[4] = bayer[bayer_step+2];
                        dst[5] = (T)t1;
                        dst[6] = alpha; // alpha channel
                    }
                }
                else // if current row contains Blue pixels
                {
                    for( ; bayer <= bayer_end - 2; bayer += 2, dst += dcn2 )
                    {
                        t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] +
                              bayer[bayer_step*2+2] + 2) >> 2;
                        t1 = (bayer[1] + bayer[bayer_step] +
                              bayer[bayer_step+2] + bayer[bayer_step*2+1]+2) >> 2;
                        dst[-1] = bayer[bayer_step+1];
                        dst[0] = (T)t1;
                        dst[1] = (T)t0;
                        dst[2] = alpha; // alpha channel

                        t0 = (bayer[2] + bayer[bayer_step*2+2] + 1) >> 1;
                        t1 = (bayer[bayer_step+1] + bayer[bayer_step+3] + 1) >> 1;
                        dst[3] = (T)t1;
                        dst[4] = bayer[bayer_step+2];
                        dst[5] = (T)t0;
                        dst[6] = alpha; // alpha channel
                    }
                }
            }

            // if skip one pixel at the end of row
            if( bayer < bayer_end )
            {
                t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] +
                      bayer[bayer_step*2+2] + 2) >> 2;
                t1 = (bayer[1] + bayer[bayer_step] +
                      bayer[bayer_step+2] + bayer[bayer_step*2+1]+2) >> 2;
                dst[-blue] = (T)t0;
                dst[0] = (T)t1;
                dst[blue] = bayer[bayer_step+1];
                if (dcn == 4)
                    dst[2] = alpha; // alpha channel
                bayer++;
                dst += dcn;
            }

            // fill the last and the first pixels of row accordingly
            if (dcn == 3)
            {
                dst0[-4] = dst0[-1];
                dst0[-3] = dst0[0];
                dst0[-2] = dst0[1];
                dst0[size.width*dcn-1] = dst0[size.width*dcn-4];
                dst0[size.width*dcn] = dst0[size.width*dcn-3];
                dst0[size.width*dcn+1] = dst0[size.width*dcn-2];
            }
            else
            {
                dst0[-5] = dst0[-1];
                dst0[-4] = dst0[0];
                dst0[-3] = dst0[1];
                dst0[-2] = dst0[2]; // alpha channel
                dst0[size.width*dcn-1] = dst0[size.width*dcn-5];
                dst0[size.width*dcn] = dst0[size.width*dcn-4];
                dst0[size.width*dcn+1] = dst0[size.width*dcn-3];
                dst0[size.width*dcn+2] = dst0[size.width*dcn-2]; // alpha channel
            }

            blue = -blue;
            start_with_green = !start_with_green;
        }
    }

private:
    Mat srcmat;
    Mat dstmat;
    int Start_with_green, Blue;
    Size size;
};

template<typename T, class SIMDInterpolator>
static void Bayer2RGB_( const Mat& srcmat, Mat& dstmat, int code )
{
    int dst_step = (int)(dstmat.step/sizeof(T));
    Size size = srcmat.size();
    int blue = code == CV_BayerBG2BGR || code == CV_BayerGB2BGR ? -1 : 1;
    int start_with_green = code == CV_BayerGB2BGR || code == CV_BayerGR2BGR;

    int dcn = dstmat.channels();
    size.height -= 2;
    size.width -= 2;

    if (size.height > 0)
    {
        Range range(0, size.height);
        Bayer2RGB_Invoker<T, SIMDInterpolator> invoker(srcmat, dstmat, start_with_green, blue, size);
        parallel_for_(range, invoker, dstmat.total()/static_cast<double>(1<<16));
    }

    // filling the first and the last rows
    size = dstmat.size();
    T* dst0 = dstmat.ptr<T>();
    if( size.height > 2 )
        for( int i = 0; i < size.width*dcn; i++ )
        {
            dst0[i] = dst0[i + dst_step];
            dst0[i + (size.height-1)*dst_step] = dst0[i + (size.height-2)*dst_step];
        }
    else
        for( int i = 0; i < size.width*dcn; i++ )
            dst0[i] = dst0[i + (size.height-1)*dst_step] = 0;
}


/////////////////// Demosaicing using Variable Number of Gradients ///////////////////////

static void Bayer2RGB_VNG_8u( const Mat& srcmat, Mat& dstmat, int code )
{
    const uchar* bayer = srcmat.ptr();
    int bstep = (int)srcmat.step;
    uchar* dst = dstmat.ptr();
    int dststep = (int)dstmat.step;
    Size size = srcmat.size();

    int blueIdx = code == CV_BayerBG2BGR_VNG || code == CV_BayerGB2BGR_VNG ? 0 : 2;
    bool greenCell0 = code != CV_BayerBG2BGR_VNG && code != CV_BayerRG2BGR_VNG;

    // for too small images use the simple interpolation algorithm
    if( MIN(size.width, size.height) < 8 )
    {
        Bayer2RGB_<uchar, SIMDBayerInterpolator_8u>( srcmat, dstmat, code );
        return;
    }

    const int brows = 3, bcn = 7;
    int N = size.width, N2 = N*2, N3 = N*3, N4 = N*4, N5 = N*5, N6 = N*6, N7 = N*7;
    int i, bufstep = N7*bcn;
    cv::AutoBuffer<ushort> _buf(bufstep*brows);
    ushort* buf = (ushort*)_buf;

    bayer += bstep*2;

    for( int y = 2; y < size.height - 4; y++ )
    {
        uchar* dstrow = dst + dststep*y + 6;
        const uchar* srow;

        for( int dy = (y == 2 ? -1 : 1); dy <= 1; dy++ )
        {
            ushort* brow = buf + ((y + dy - 1)%brows)*bufstep + 1;
            srow = bayer + (y+dy)*bstep + 1;

            for( i = 0; i < bcn; i++ )
                brow[N*i-1] = brow[(N-2) + N*i] = 0;

            i = 1;

            for( ; i < N-1; i++, srow++, brow++ )
            {
                brow[0] = (ushort)(std::abs(srow[-1-bstep] - srow[-1+bstep]) +
                                   std::abs(srow[-bstep] - srow[+bstep])*2 +
                                   std::abs(srow[1-bstep] - srow[1+bstep]));
                brow[N] = (ushort)(std::abs(srow[-1-bstep] - srow[1-bstep]) +
                                   std::abs(srow[-1] - srow[1])*2 +
                                   std::abs(srow[-1+bstep] - srow[1+bstep]));
                brow[N2] = (ushort)(std::abs(srow[+1-bstep] - srow[-1+bstep])*2);
                brow[N3] = (ushort)(std::abs(srow[-1-bstep] - srow[1+bstep])*2);
                brow[N4] = (ushort)(brow[N2] + std::abs(srow[-bstep] - srow[-1]) +
                                    std::abs(srow[+bstep] - srow[1]));
                brow[N5] = (ushort)(brow[N3] + std::abs(srow[-bstep] - srow[1]) +
                                    std::abs(srow[+bstep] - srow[-1]));
                brow[N6] = (ushort)((srow[-bstep] + srow[-1] + srow[1] + srow[+bstep])>>1);
            }
        }

        const ushort* brow0 = buf + ((y - 2) % brows)*bufstep + 2;
        const ushort* brow1 = buf + ((y - 1) % brows)*bufstep + 2;
        const ushort* brow2 = buf + (y % brows)*bufstep + 2;
        static const float scale[] = { 0.f, 0.5f, 0.25f, 0.1666666666667f, 0.125f, 0.1f, 0.08333333333f, 0.0714286f, 0.0625f };
        srow = bayer + y*bstep + 2;
        bool greenCell = greenCell0;

        i = 2;
        int limit = N - 2;
        do
        {
            for( ; i < limit; i++, srow++, brow0++, brow1++, brow2++, dstrow += 3 )
            {
                int gradN = brow0[0] + brow1[0];
                int gradS = brow1[0] + brow2[0];
                int gradW = brow1[N-1] + brow1[N];
                int gradE = brow1[N] + brow1[N+1];
                int minGrad = std::min(std::min(std::min(gradN, gradS), gradW), gradE);
                int maxGrad = std::max(std::max(std::max(gradN, gradS), gradW), gradE);
                int R, G, B;

                if( !greenCell )
                {
                    int gradNE = brow0[N4+1] + brow1[N4];
                    int gradSW = brow1[N4] + brow2[N4-1];
                    int gradNW = brow0[N5-1] + brow1[N5];
                    int gradSE = brow1[N5] + brow2[N5+1];

                    minGrad = std::min(std::min(std::min(std::min(minGrad, gradNE), gradSW), gradNW), gradSE);
                    maxGrad = std::max(std::max(std::max(std::max(maxGrad, gradNE), gradSW), gradNW), gradSE);
                    int T = minGrad + MAX(maxGrad/2, 1);

                    int Rs = 0, Gs = 0, Bs = 0, ng = 0;
                    if( gradN < T )
                    {
                        Rs += srow[-bstep*2] + srow[0];
                        Gs += srow[-bstep]*2;
                        Bs += srow[-bstep-1] + srow[-bstep+1];
                        ng++;
                    }
                    if( gradS < T )
                    {
                        Rs += srow[bstep*2] + srow[0];
                        Gs += srow[bstep]*2;
                        Bs += srow[bstep-1] + srow[bstep+1];
                        ng++;
                    }
                    if( gradW < T )
                    {
                        Rs += srow[-2] + srow[0];
                        Gs += srow[-1]*2;
                        Bs += srow[-bstep-1] + srow[bstep-1];
                        ng++;
                    }
                    if( gradE < T )
                    {
                        Rs += srow[2] + srow[0];
                        Gs += srow[1]*2;
                        Bs += srow[-bstep+1] + srow[bstep+1];
                        ng++;
                    }
                    if( gradNE < T )
                    {
                        Rs += srow[-bstep*2+2] + srow[0];
                        Gs += brow0[N6+1];
                        Bs += srow[-bstep+1]*2;
                        ng++;
                    }
                    if( gradSW < T )
                    {
                        Rs += srow[bstep*2-2] + srow[0];
                        Gs += brow2[N6-1];
                        Bs += srow[bstep-1]*2;
                        ng++;
                    }
                    if( gradNW < T )
                    {
                        Rs += srow[-bstep*2-2] + srow[0];
                        Gs += brow0[N6-1];
                        Bs += srow[-bstep+1]*2;
                        ng++;
                    }
                    if( gradSE < T )
                    {
                        Rs += srow[bstep*2+2] + srow[0];
                        Gs += brow2[N6+1];
                        Bs += srow[-bstep+1]*2;
                        ng++;
                    }
                    R = srow[0];
                    G = R + cvRound((Gs - Rs)*scale[ng]);
                    B = R + cvRound((Bs - Rs)*scale[ng]);
                }
                else
                {
                    int gradNE = brow0[N2] + brow0[N2+1] + brow1[N2] + brow1[N2+1];
                    int gradSW = brow1[N2] + brow1[N2-1] + brow2[N2] + brow2[N2-1];
                    int gradNW = brow0[N3] + brow0[N3-1] + brow1[N3] + brow1[N3-1];
                    int gradSE = brow1[N3] + brow1[N3+1] + brow2[N3] + brow2[N3+1];

                    minGrad = std::min(std::min(std::min(std::min(minGrad, gradNE), gradSW), gradNW), gradSE);
                    maxGrad = std::max(std::max(std::max(std::max(maxGrad, gradNE), gradSW), gradNW), gradSE);
                    int T = minGrad + MAX(maxGrad/2, 1);

                    int Rs = 0, Gs = 0, Bs = 0, ng = 0;
                    if( gradN < T )
                    {
                        Rs += srow[-bstep*2-1] + srow[-bstep*2+1];
                        Gs += srow[-bstep*2] + srow[0];
                        Bs += srow[-bstep]*2;
                        ng++;
                    }
                    if( gradS < T )
                    {
                        Rs += srow[bstep*2-1] + srow[bstep*2+1];
                        Gs += srow[bstep*2] + srow[0];
                        Bs += srow[bstep]*2;
                        ng++;
                    }
                    if( gradW < T )
                    {
                        Rs += srow[-1]*2;
                        Gs += srow[-2] + srow[0];
                        Bs += srow[-bstep-2]+srow[bstep-2];
                        ng++;
                    }
                    if( gradE < T )
                    {
                        Rs += srow[1]*2;
                        Gs += srow[2] + srow[0];
                        Bs += srow[-bstep+2]+srow[bstep+2];
                        ng++;
                    }
                    if( gradNE < T )
                    {
                        Rs += srow[-bstep*2+1] + srow[1];
                        Gs += srow[-bstep+1]*2;
                        Bs += srow[-bstep] + srow[-bstep+2];
                        ng++;
                    }
                    if( gradSW < T )
                    {
                        Rs += srow[bstep*2-1] + srow[-1];
                        Gs += srow[bstep-1]*2;
                        Bs += srow[bstep] + srow[bstep-2];
                        ng++;
                    }
                    if( gradNW < T )
                    {
                        Rs += srow[-bstep*2-1] + srow[-1];
                        Gs += srow[-bstep-1]*2;
                        Bs += srow[-bstep-2]+srow[-bstep];
                        ng++;
                    }
                    if( gradSE < T )
                    {
                        Rs += srow[bstep*2+1] + srow[1];
                        Gs += srow[bstep+1]*2;
                        Bs += srow[bstep+2]+srow[bstep];
                        ng++;
                    }
                    G = srow[0];
                    R = G + cvRound((Rs - Gs)*scale[ng]);
                    B = G + cvRound((Bs - Gs)*scale[ng]);
                }
                dstrow[blueIdx] = cv::saturate_cast<uchar>(B);
                dstrow[1] = cv::saturate_cast<uchar>(G);
                dstrow[blueIdx^2] = cv::saturate_cast<uchar>(R);
                greenCell = !greenCell;
            }

            limit = N - 2;
        }
        while( i < N - 2 );

        for( i = 0; i < 6; i++ )
        {
            dst[dststep*y + 5 - i] = dst[dststep*y + 8 - i];
            dst[dststep*y + (N - 2)*3 + i] = dst[dststep*y + (N - 3)*3 + i];
        }

        greenCell0 = !greenCell0;
        blueIdx ^= 2;
    }

    for( i = 0; i < size.width*3; i++ )
    {
        dst[i] = dst[i + dststep] = dst[i + dststep*2];
        dst[i + dststep*(size.height-4)] =
        dst[i + dststep*(size.height-3)] =
        dst[i + dststep*(size.height-2)] =
        dst[i + dststep*(size.height-1)] = dst[i + dststep*(size.height-5)];
    }
}

//////////////////////////////// Edge-Aware Demosaicing //////////////////////////////////

template <typename T, typename SIMDInterpolator>
class Bayer2RGB_EdgeAware_T_Invoker :
    public cv::ParallelLoopBody
{
public:
    Bayer2RGB_EdgeAware_T_Invoker(const Mat& _src, Mat& _dst, const Size& _size,
        int _blue, int _start_with_green) :
        ParallelLoopBody(),
        src(_src), dst(_dst), size(_size), Blue(_blue), Start_with_green(_start_with_green)
    {
    }

    virtual void operator()(const Range& range) const
    {
        int dcn = dst.channels();
        int dcn2 = dcn<<1;
        int start_with_green = Start_with_green, blue = Blue;
        int sstep = int(src.step / src.elemSize1()), dstep = int(dst.step / dst.elemSize1());
        SIMDInterpolator vecOp;

        const T* S = src.ptr<T>(range.start + 1) + 1;
        T* D = reinterpret_cast<T*>(dst.data + (range.start + 1) * dst.step) + dcn;

        if (range.start % 2)
        {
            start_with_green ^= 1;
            blue ^= 1;
        }

        // to BGR
        for (int y = range.start; y < range.end; ++y)
        {
            int x = 1;
            if (start_with_green)
            {
                D[blue<<1] = (S[-sstep] + S[sstep]) >> 1;
                D[1] = S[0];
                D[2-(blue<<1)] = (S[-1] + S[1]) >> 1;
                D += dcn;
                ++S;
                ++x;
            }

            int delta = vecOp.bayer2RGB_EA(S - sstep - 1, sstep, D, size.width, blue);
            x += delta;
            S += delta;
            D += dcn * delta;

            if (blue)
                for (; x < size.width; x += 2, S += 2, D += dcn2)
                {
                    D[0] = S[0];
                    D[1] = (std::abs(S[-1] - S[1]) > std::abs(S[sstep] - S[-sstep]) ? (S[sstep] + S[-sstep] + 1) : (S[-1] + S[1] + 1)) >> 1;
                    D[2] = (S[-sstep-1] + S[-sstep+1] + S[sstep-1] + S[sstep+1]) >> 2;

                    D[3] = (S[0] + S[2] + 1) >> 1;
                    D[4] = S[1];
                    D[5] = (S[-sstep+1] + S[sstep+1] + 1) >> 1;
                }
            else
                for (; x < size.width; x += 2, S += 2, D += dcn2)
                {
                    D[0] = (S[-sstep-1] + S[-sstep+1] + S[sstep-1] + S[sstep+1] + 2) >> 2;
                    D[1] = (std::abs(S[-1] - S[1]) > std::abs(S[sstep] - S[-sstep]) ? (S[sstep] + S[-sstep] + 1) : (S[-1] + S[1] + 1)) >> 1;
                    D[2] = S[0];

                    D[3] = (S[-sstep+1] + S[sstep+1] + 1) >> 1;
                    D[4] = S[1];
                    D[5] = (S[0] + S[2] + 1) >> 1;
                }

            if (x <= size.width)
            {
                D[blue<<1] = (S[-sstep-1] + S[-sstep+1] + S[sstep-1] + S[sstep+1] + 2) >> 2;
                D[1] = (std::abs(S[-1] - S[1]) > std::abs(S[sstep] - S[-sstep]) ? (S[sstep] + S[-sstep] + 1) : (S[-1] + S[1] + 1)) >> 1;
                D[2-(blue<<1)] = S[0];
                D += dcn;
                ++S;
            }

            for (int i = 0; i < dcn; ++i)
            {
                D[i] = D[-dcn + i];
                D[-dstep+dcn+i] = D[-dstep+(dcn<<1)+i];
            }

            start_with_green ^= 1;
            blue ^= 1;
            S += 2;
            D += dcn2;
        }
    }

private:
    Mat src;
    Mat dst;
    Size size;
    int Blue, Start_with_green;
};

template <typename T, typename SIMDInterpolator>
static void Bayer2RGB_EdgeAware_T(const Mat& src, Mat& dst, int code)
{
    Size size = src.size();

    // for small sizes
    if (size.width <= 2 || size.height <= 2)
    {
        dst = Scalar::all(0);
        return;
    }

    size.width -= 2;
    size.height -= 2;

    int start_with_green = code == CV_BayerGB2BGR_EA || code == CV_BayerGR2BGR_EA ? 1 : 0;
    int blue = code == CV_BayerGB2BGR_EA || code == CV_BayerBG2BGR_EA ? 1 : 0;

    if (size.height > 0)
    {
        Bayer2RGB_EdgeAware_T_Invoker<T, SIMDInterpolator> invoker(src, dst, size, blue, start_with_green);
        Range range(0, size.height);
        parallel_for_(range, invoker, dst.total()/static_cast<double>(1<<16));
    }
    size = dst.size();
    size.width *= dst.channels();
    size_t dstep = dst.step / dst.elemSize1();
    T* firstRow = dst.ptr<T>();
    T* lastRow = dst.ptr<T>() + (size.height-1) * dstep;

    if (size.height > 2)
    {
        for (int x = 0; x < size.width; ++x)
        {
            firstRow[x] = (firstRow+dstep)[x];
            lastRow[x] = (lastRow-dstep)[x];
        }
    }
    else
        for (int x = 0; x < size.width; ++x)
            firstRow[x] = lastRow[x] = 0;
}

} // end namespace cv

//////////////////////////////////////////////////////////////////////////////////////////
//                           The main Demosaicing function                              //
//////////////////////////////////////////////////////////////////////////////////////////

void cv::demosaicing(InputArray _src, OutputArray _dst, int code, int dcn)
{
    CV_INSTRUMENT_REGION()

    Mat src = _src.getMat(), dst;
    Size sz = src.size();
    int scn = src.channels(), depth = src.depth();

    CV_Assert(depth == CV_8U || depth == CV_16U);
    CV_Assert(!src.empty());

    switch (code)
    {
    case CV_BayerBG2GRAY: case CV_BayerGB2GRAY: case CV_BayerRG2GRAY: case CV_BayerGR2GRAY:
        if (dcn <= 0)
            dcn = 1;
        CV_Assert( scn == 1 && dcn == 1 );

        _dst.create(sz, CV_MAKETYPE(depth, dcn));
        dst = _dst.getMat();

        if( depth == CV_8U )
            Bayer2Gray_<uchar, SIMDBayerInterpolator_8u>(src, dst, code);
        else if( depth == CV_16U )
            Bayer2Gray_<ushort, SIMDBayerStubInterpolator_<ushort> >(src, dst, code);
        else
            CV_Error(CV_StsUnsupportedFormat, "Bayer->Gray demosaicing only supports 8u and 16u types");
        break;

    case CV_BayerBG2BGR: case CV_BayerGB2BGR: case CV_BayerRG2BGR: case CV_BayerGR2BGR:
    case CV_BayerBG2BGR_VNG: case CV_BayerGB2BGR_VNG: case CV_BayerRG2BGR_VNG: case CV_BayerGR2BGR_VNG:
        {
            if (dcn <= 0)
                dcn = 3;
            CV_Assert( scn == 1 && (dcn == 3 || dcn == 4) );

            _dst.create(sz, CV_MAKE_TYPE(depth, dcn));
            Mat dst_ = _dst.getMat();

            if( code == CV_BayerBG2BGR || code == CV_BayerGB2BGR ||
                code == CV_BayerRG2BGR || code == CV_BayerGR2BGR )
            {
                if( depth == CV_8U )
                    Bayer2RGB_<uchar, SIMDBayerInterpolator_8u>(src, dst_, code);
                else if( depth == CV_16U )
                    Bayer2RGB_<ushort, SIMDBayerStubInterpolator_<ushort> >(src, dst_, code);
                else
                    CV_Error(CV_StsUnsupportedFormat, "Bayer->RGB demosaicing only supports 8u and 16u types");
            }
            else
            {
                CV_Assert( depth == CV_8U );
                Bayer2RGB_VNG_8u(src, dst_, code);
            }
        }
        break;

    case CV_BayerBG2BGR_EA: case CV_BayerGB2BGR_EA: case CV_BayerRG2BGR_EA: case CV_BayerGR2BGR_EA:
        if (dcn <= 0)
            dcn = 3;

        CV_Assert(scn == 1 && dcn == 3);
        _dst.create(sz, CV_MAKETYPE(depth, dcn));
        dst = _dst.getMat();

        if (depth == CV_8U)
            Bayer2RGB_EdgeAware_T<uchar, SIMDBayerInterpolator_8u>(src, dst, code);
        else if (depth == CV_16U)
            Bayer2RGB_EdgeAware_T<ushort, SIMDBayerStubInterpolator_<ushort> >(src, dst, code);
        else
            CV_Error(CV_StsUnsupportedFormat, "Bayer->RGB Edge-Aware demosaicing only currently supports 8u and 16u types");

        break;

    default:
        CV_Error( CV_StsBadFlag, "Unknown / unsupported color conversion code" );
    }
}
