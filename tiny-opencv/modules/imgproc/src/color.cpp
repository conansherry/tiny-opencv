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

/********************************* COPYRIGHT NOTICE *******************************\
  The function for RGB to Lab conversion is based on the MATLAB script
  RGB2Lab.m translated by Mark Ruzon from C code by Yossi Rubner, 23 September 1997.
  See the page [http://vision.stanford.edu/~ruzon/software/rgblab.html]
\**********************************************************************************/

/********************************* COPYRIGHT NOTICE *******************************\
  Original code for Bayer->BGR/RGB conversion is provided by Dirk Schaefer
  from MD-Mathematische Dienste GmbH. Below is the copyright notice:

    IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
    By downloading, copying, installing or using the software you agree
    to this license. If you do not agree to this license, do not download,
    install, copy or use the software.

    Contributors License Agreement:

      Copyright (c) 2002,
      MD-Mathematische Dienste GmbH
      Im Defdahl 5-10
      44141 Dortmund
      Germany
      www.md-it.de

    Redistribution and use in source and binary forms,
    with or without modification, are permitted provided
    that the following conditions are met:

    Redistributions of source code must retain
    the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
    The name of Contributor may not be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
    OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
    THE POSSIBILITY OF SUCH DAMAGE.
\**********************************************************************************/

#include "precomp.hpp"
#include <limits>
#include "hal_replacement.hpp"

#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))

#if defined (HAVE_IPP) && (IPP_VERSION_X100 >= 700)
#define MAX_IPP8u   255
#define MAX_IPP16u  65535
#define MAX_IPP32f  1.0
#endif

namespace cv
{
//constants for conversion from/to RGB and Gray, YUV, YCrCb according to BT.601
const float B2YF = 0.114f;
const float G2YF = 0.587f;
const float R2YF = 0.299f;
//to YCbCr
const float YCBF = 0.564f; // == 1/2/(1-B2YF)
const float YCRF = 0.713f; // == 1/2/(1-R2YF)
const int YCBI = 9241;  // == YCBF*16384
const int YCRI = 11682; // == YCRF*16384
//to YUV
const float B2UF = 0.492f;
const float R2VF = 0.877f;
const int B2UI = 8061;  // == B2UF*16384
const int R2VI = 14369; // == R2VF*16384
//from YUV
const float U2BF = 2.032f;
const float U2GF = -0.395f;
const float V2GF = -0.581f;
const float V2RF = 1.140f;
const int U2BI = 33292;
const int U2GI = -6472;
const int V2GI = -9519;
const int V2RI = 18678;
//from YCrCb
const float CB2BF = 1.773f;
const float CB2GF = -0.344f;
const float CR2GF = -0.714f;
const float CR2RF = 1.403f;
const int CB2BI = 29049;
const int CB2GI = -5636;
const int CR2GI = -11698;
const int CR2RI = 22987;


// computes cubic spline coefficients for a function: (xi=i, yi=f[i]), i=0..n
template<typename _Tp> static void splineBuild(const _Tp* f, int n, _Tp* tab)
{
    _Tp cn = 0;
    int i;
    tab[0] = tab[1] = (_Tp)0;

    for(i = 1; i < n-1; i++)
    {
        _Tp t = 3*(f[i+1] - 2*f[i] + f[i-1]);
        _Tp l = 1/(4 - tab[(i-1)*4]);
        tab[i*4] = l; tab[i*4+1] = (t - tab[(i-1)*4+1])*l;
    }

    for(i = n-1; i >= 0; i--)
    {
        _Tp c = tab[i*4+1] - tab[i*4]*cn;
        _Tp b = f[i+1] - f[i] - (cn + c*2)*(_Tp)0.3333333333333333;
        _Tp d = (cn - c)*(_Tp)0.3333333333333333;
        tab[i*4] = f[i]; tab[i*4+1] = b;
        tab[i*4+2] = c; tab[i*4+3] = d;
        cn = c;
    }
}

// interpolates value of a function at x, 0 <= x <= n using a cubic spline.
template<typename _Tp> static inline _Tp splineInterpolate(_Tp x, const _Tp* tab, int n)
{
    // don't touch this function without urgent need - some versions of gcc fail to inline it correctly
    int ix = std::min(std::max(int(x), 0), n-1);
    x -= ix;
    tab += ix*4;
    return ((tab[3]*x + tab[2])*x + tab[1])*x + tab[0];
}

#if CV_NEON
template<typename _Tp> static inline void splineInterpolate(float32x4_t& v_x, const _Tp* tab, int n)
{
    int32x4_t v_ix = vcvtq_s32_f32(vminq_f32(vmaxq_f32(v_x, vdupq_n_f32(0)), vdupq_n_f32(n - 1)));
    v_x = vsubq_f32(v_x, vcvtq_f32_s32(v_ix));
    v_ix = vshlq_n_s32(v_ix, 2);

    int CV_DECL_ALIGNED(16) ix[4];
    vst1q_s32(ix, v_ix);

    float32x4_t v_tab0 = vld1q_f32(tab + ix[0]);
    float32x4_t v_tab1 = vld1q_f32(tab + ix[1]);
    float32x4_t v_tab2 = vld1q_f32(tab + ix[2]);
    float32x4_t v_tab3 = vld1q_f32(tab + ix[3]);

    float32x4x2_t v01 = vtrnq_f32(v_tab0, v_tab1);
    float32x4x2_t v23 = vtrnq_f32(v_tab2, v_tab3);

    v_tab0 = vcombine_f32(vget_low_f32(v01.val[0]), vget_low_f32(v23.val[0]));
    v_tab1 = vcombine_f32(vget_low_f32(v01.val[1]), vget_low_f32(v23.val[1]));
    v_tab2 = vcombine_f32(vget_high_f32(v01.val[0]), vget_high_f32(v23.val[0]));
    v_tab3 = vcombine_f32(vget_high_f32(v01.val[1]), vget_high_f32(v23.val[1]));

    v_x = vmlaq_f32(v_tab0, vmlaq_f32(v_tab1, vmlaq_f32(v_tab2, v_tab3, v_x), v_x), v_x);
}
#endif

template<typename _Tp> struct ColorChannel
{
    typedef float worktype_f;
    static _Tp max() { return std::numeric_limits<_Tp>::max(); }
    static _Tp half() { return (_Tp)(max()/2 + 1); }
};

template<> struct ColorChannel<float>
{
    typedef float worktype_f;
    static float max() { return 1.f; }
    static float half() { return 0.5f; }
};

/*template<> struct ColorChannel<double>
{
    typedef double worktype_f;
    static double max() { return 1.; }
    static double half() { return 0.5; }
};*/


///////////////////////////// Top-level template function ////////////////////////////////

template <typename Cvt>
class CvtColorLoop_Invoker : public ParallelLoopBody
{
    typedef typename Cvt::channel_type _Tp;
public:

    CvtColorLoop_Invoker(const uchar * src_data_, size_t src_step_, uchar * dst_data_, size_t dst_step_, int width_, const Cvt& _cvt) :
        ParallelLoopBody(), src_data(src_data_), src_step(src_step_), dst_data(dst_data_), dst_step(dst_step_),
        width(width_), cvt(_cvt)
    {
    }

    virtual void operator()(const Range& range) const
    {
        const uchar* yS = src_data + static_cast<size_t>(range.start) * src_step;
        uchar* yD = dst_data + static_cast<size_t>(range.start) * dst_step;

        for( int i = range.start; i < range.end; ++i, yS += src_step, yD += dst_step )
            cvt(reinterpret_cast<const _Tp*>(yS), reinterpret_cast<_Tp*>(yD), width);
    }

private:
    const uchar * src_data;
    size_t src_step;
    uchar * dst_data;
    size_t dst_step;
    int width;
    const Cvt& cvt;

    const CvtColorLoop_Invoker& operator= (const CvtColorLoop_Invoker&);
};

template <typename Cvt>
void CvtColorLoop(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, const Cvt& cvt)
{
    parallel_for_(Range(0, height),
                  CvtColorLoop_Invoker<Cvt>(src_data, src_step, dst_data, dst_step, width, cvt),
                  (width * height) / static_cast<double>(1<<16));
}

////////////////// Various 3/4-channel to 3/4-channel RGB transformations /////////////////

template<typename _Tp> struct RGB2RGB
{
    typedef _Tp channel_type;

    RGB2RGB(int _srccn, int _dstcn, int _blueIdx) : srccn(_srccn), dstcn(_dstcn), blueIdx(_blueIdx) {}
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn, dcn = dstcn, bidx = blueIdx;
        if( dcn == 3 )
        {
            n *= 3;
            for( int i = 0; i < n; i += 3, src += scn )
            {
                _Tp t0 = src[bidx], t1 = src[1], t2 = src[bidx ^ 2];
                dst[i] = t0; dst[i+1] = t1; dst[i+2] = t2;
            }
        }
        else if( scn == 3 )
        {
            n *= 3;
            _Tp alpha = ColorChannel<_Tp>::max();
            for( int i = 0; i < n; i += 3, dst += 4 )
            {
                _Tp t0 = src[i], t1 = src[i+1], t2 = src[i+2];
                dst[bidx] = t0; dst[1] = t1; dst[bidx^2] = t2; dst[3] = alpha;
            }
        }
        else
        {
            n *= 4;
            for( int i = 0; i < n; i += 4 )
            {
                _Tp t0 = src[i], t1 = src[i+1], t2 = src[i+2], t3 = src[i+3];
                dst[i] = t2; dst[i+1] = t1; dst[i+2] = t0; dst[i+3] = t3;
            }
        }
    }

    int srccn, dstcn, blueIdx;
};

#if CV_NEON

template<> struct RGB2RGB<uchar>
{
    typedef uchar channel_type;

    RGB2RGB(int _srccn, int _dstcn, int _blueIdx) :
        srccn(_srccn), dstcn(_dstcn), blueIdx(_blueIdx)
    {
        v_alpha = vdupq_n_u8(ColorChannel<uchar>::max());
        v_alpha2 = vget_low_u8(v_alpha);
    }

    void operator()(const uchar * src, uchar * dst, int n) const
    {
        int scn = srccn, dcn = dstcn, bidx = blueIdx, i = 0;
        if (dcn == 3)
        {
            n *= 3;
            if (scn == 3)
            {
                for ( ; i <= n - 48; i += 48, src += 48 )
                {
                    uint8x16x3_t v_src = vld3q_u8(src), v_dst;
                    v_dst.val[0] = v_src.val[bidx];
                    v_dst.val[1] = v_src.val[1];
                    v_dst.val[2] = v_src.val[bidx ^ 2];
                    vst3q_u8(dst + i, v_dst);
                }
                for ( ; i <= n - 24; i += 24, src += 24 )
                {
                    uint8x8x3_t v_src = vld3_u8(src), v_dst;
                    v_dst.val[0] = v_src.val[bidx];
                    v_dst.val[1] = v_src.val[1];
                    v_dst.val[2] = v_src.val[bidx ^ 2];
                    vst3_u8(dst + i, v_dst);
                }
                for ( ; i < n; i += 3, src += 3 )
                {
                    uchar t0 = src[bidx], t1 = src[1], t2 = src[bidx ^ 2];
                    dst[i] = t0; dst[i+1] = t1; dst[i+2] = t2;
                }
            }
            else
            {
                for ( ; i <= n - 48; i += 48, src += 64 )
                {
                    uint8x16x4_t v_src = vld4q_u8(src);
                    uint8x16x3_t v_dst;
                    v_dst.val[0] = v_src.val[bidx];
                    v_dst.val[1] = v_src.val[1];
                    v_dst.val[2] = v_src.val[bidx ^ 2];
                    vst3q_u8(dst + i, v_dst);
                }
                for ( ; i <= n - 24; i += 24, src += 32 )
                {
                    uint8x8x4_t v_src = vld4_u8(src);
                    uint8x8x3_t v_dst;
                    v_dst.val[0] = v_src.val[bidx];
                    v_dst.val[1] = v_src.val[1];
                    v_dst.val[2] = v_src.val[bidx ^ 2];
                    vst3_u8(dst + i, v_dst);
                }
                for ( ; i < n; i += 3, src += 4 )
                {
                    uchar t0 = src[bidx], t1 = src[1], t2 = src[bidx ^ 2];
                    dst[i] = t0; dst[i+1] = t1; dst[i+2] = t2;
                }
            }
        }
        else if (scn == 3)
        {
            n *= 3;
            for ( ; i <= n - 48; i += 48, dst += 64 )
            {
                uint8x16x3_t v_src = vld3q_u8(src + i);
                uint8x16x4_t v_dst;
                v_dst.val[bidx] = v_src.val[0];
                v_dst.val[1] = v_src.val[1];
                v_dst.val[bidx ^ 2] = v_src.val[2];
                v_dst.val[3] = v_alpha;
                vst4q_u8(dst, v_dst);
            }
            for ( ; i <= n - 24; i += 24, dst += 32 )
            {
                uint8x8x3_t v_src = vld3_u8(src + i);
                uint8x8x4_t v_dst;
                v_dst.val[bidx] = v_src.val[0];
                v_dst.val[1] = v_src.val[1];
                v_dst.val[bidx ^ 2] = v_src.val[2];
                v_dst.val[3] = v_alpha2;
                vst4_u8(dst, v_dst);
            }
            uchar alpha = ColorChannel<uchar>::max();
            for (; i < n; i += 3, dst += 4 )
            {
                uchar t0 = src[i], t1 = src[i+1], t2 = src[i+2];
                dst[bidx] = t0; dst[1] = t1; dst[bidx^2] = t2; dst[3] = alpha;
            }
        }
        else
        {
            n *= 4;
            for ( ; i <= n - 64; i += 64 )
            {
                uint8x16x4_t v_src = vld4q_u8(src + i), v_dst;
                v_dst.val[0] = v_src.val[2];
                v_dst.val[1] = v_src.val[1];
                v_dst.val[2] = v_src.val[0];
                v_dst.val[3] = v_src.val[3];
                vst4q_u8(dst + i, v_dst);
            }
            for ( ; i <= n - 32; i += 32 )
            {
                uint8x8x4_t v_src = vld4_u8(src + i), v_dst;
                v_dst.val[0] = v_src.val[2];
                v_dst.val[1] = v_src.val[1];
                v_dst.val[2] = v_src.val[0];
                v_dst.val[3] = v_src.val[3];
                vst4_u8(dst + i, v_dst);
            }
            for ( ; i < n; i += 4)
            {
                uchar t0 = src[i], t1 = src[i+1], t2 = src[i+2], t3 = src[i+3];
                dst[i] = t2; dst[i+1] = t1; dst[i+2] = t0; dst[i+3] = t3;
            }
        }
    }

    int srccn, dstcn, blueIdx;

    uint8x16_t v_alpha;
    uint8x8_t v_alpha2;
};

#endif

/////////// Transforming 16-bit (565 or 555) RGB to/from 24/32-bit (888[8]) RGB //////////

struct RGB5x52RGB
{
    typedef uchar channel_type;

    RGB5x52RGB(int _dstcn, int _blueIdx, int _greenBits)
        : dstcn(_dstcn), blueIdx(_blueIdx), greenBits(_greenBits)
    {
        #if CV_NEON
        v_n3 = vdupq_n_u16(~3);
        v_n7 = vdupq_n_u16(~7);
        v_255 = vdupq_n_u8(255);
        v_0 = vdupq_n_u8(0);
        v_mask = vdupq_n_u16(0x8000);
        #endif
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx, i = 0;
        if( greenBits == 6 )
        {
            #if CV_NEON
            for ( ; i <= n - 16; i += 16, dst += dcn * 16)
            {
                uint16x8_t v_src0 = vld1q_u16((const ushort *)src + i), v_src1 = vld1q_u16((const ushort *)src + i + 8);
                uint8x16_t v_b = vcombine_u8(vmovn_u16(vshlq_n_u16(v_src0, 3)), vmovn_u16(vshlq_n_u16(v_src1, 3)));
                uint8x16_t v_g = vcombine_u8(vmovn_u16(vandq_u16(vshrq_n_u16(v_src0, 3), v_n3)),
                                             vmovn_u16(vandq_u16(vshrq_n_u16(v_src1, 3), v_n3)));
                uint8x16_t v_r = vcombine_u8(vmovn_u16(vandq_u16(vshrq_n_u16(v_src0, 8), v_n7)),
                                             vmovn_u16(vandq_u16(vshrq_n_u16(v_src1, 8), v_n7)));
                if (dcn == 3)
                {
                    uint8x16x3_t v_dst;
                    v_dst.val[bidx] = v_b;
                    v_dst.val[1] = v_g;
                    v_dst.val[bidx^2] = v_r;
                    vst3q_u8(dst, v_dst);
                }
                else
                {
                    uint8x16x4_t v_dst;
                    v_dst.val[bidx] = v_b;
                    v_dst.val[1] = v_g;
                    v_dst.val[bidx^2] = v_r;
                    v_dst.val[3] = v_255;
                    vst4q_u8(dst, v_dst);
                }
            }
            #endif
            for( ; i < n; i++, dst += dcn )
            {
                unsigned t = ((const ushort*)src)[i];
                dst[bidx] = (uchar)(t << 3);
                dst[1] = (uchar)((t >> 3) & ~3);
                dst[bidx ^ 2] = (uchar)((t >> 8) & ~7);
                if( dcn == 4 )
                    dst[3] = 255;
            }
        }
        else
        {
            #if CV_NEON
            for ( ; i <= n - 16; i += 16, dst += dcn * 16)
            {
                uint16x8_t v_src0 = vld1q_u16((const ushort *)src + i), v_src1 = vld1q_u16((const ushort *)src + i + 8);
                uint8x16_t v_b = vcombine_u8(vmovn_u16(vshlq_n_u16(v_src0, 3)), vmovn_u16(vshlq_n_u16(v_src1, 3)));
                uint8x16_t v_g = vcombine_u8(vmovn_u16(vandq_u16(vshrq_n_u16(v_src0, 2), v_n7)),
                                             vmovn_u16(vandq_u16(vshrq_n_u16(v_src1, 2), v_n7)));
                uint8x16_t v_r = vcombine_u8(vmovn_u16(vandq_u16(vshrq_n_u16(v_src0, 7), v_n7)),
                                             vmovn_u16(vandq_u16(vshrq_n_u16(v_src1, 7), v_n7)));
                if (dcn == 3)
                {
                    uint8x16x3_t v_dst;
                    v_dst.val[bidx] = v_b;
                    v_dst.val[1] = v_g;
                    v_dst.val[bidx^2] = v_r;
                    vst3q_u8(dst, v_dst);
                }
                else
                {
                    uint8x16x4_t v_dst;
                    v_dst.val[bidx] = v_b;
                    v_dst.val[1] = v_g;
                    v_dst.val[bidx^2] = v_r;
                    v_dst.val[3] = vbslq_u8(vcombine_u8(vqmovn_u16(vandq_u16(v_src0, v_mask)),
                                                        vqmovn_u16(vandq_u16(v_src1, v_mask))), v_255, v_0);
                    vst4q_u8(dst, v_dst);
                }
            }
            #endif
            for( ; i < n; i++, dst += dcn )
            {
                unsigned t = ((const ushort*)src)[i];
                dst[bidx] = (uchar)(t << 3);
                dst[1] = (uchar)((t >> 2) & ~7);
                dst[bidx ^ 2] = (uchar)((t >> 7) & ~7);
                if( dcn == 4 )
                    dst[3] = t & 0x8000 ? 255 : 0;
            }
        }
    }

    int dstcn, blueIdx, greenBits;
    #if CV_NEON
    uint16x8_t v_n3, v_n7, v_mask;
    uint8x16_t v_255, v_0;
    #endif
};


struct RGB2RGB5x5
{
    typedef uchar channel_type;

    RGB2RGB5x5(int _srccn, int _blueIdx, int _greenBits)
        : srccn(_srccn), blueIdx(_blueIdx), greenBits(_greenBits)
    {
        #if CV_NEON
        v_n3 = vdup_n_u8(~3);
        v_n7 = vdup_n_u8(~7);
        v_mask = vdupq_n_u16(0x8000);
        v_0 = vdupq_n_u16(0);
        v_full = vdupq_n_u16(0xffff);
        #endif
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int scn = srccn, bidx = blueIdx, i = 0;
        if (greenBits == 6)
        {
            if (scn == 3)
            {
                #if CV_NEON
                for ( ; i <= n - 8; i += 8, src += 24 )
                {
                    uint8x8x3_t v_src = vld3_u8(src);
                    uint16x8_t v_dst = vmovl_u8(vshr_n_u8(v_src.val[bidx], 3));
                    v_dst = vorrq_u16(v_dst, vshlq_n_u16(vmovl_u8(vand_u8(v_src.val[1], v_n3)), 3));
                    v_dst = vorrq_u16(v_dst, vshlq_n_u16(vmovl_u8(vand_u8(v_src.val[bidx^2], v_n7)), 8));
                    vst1q_u16((ushort *)dst + i, v_dst);
                }
                #endif
                for ( ; i < n; i++, src += 3 )
                    ((ushort*)dst)[i] = (ushort)((src[bidx] >> 3)|((src[1]&~3) << 3)|((src[bidx^2]&~7) << 8));
            }
            else
            {
                #if CV_NEON
                for ( ; i <= n - 8; i += 8, src += 32 )
                {
                    uint8x8x4_t v_src = vld4_u8(src);
                    uint16x8_t v_dst = vmovl_u8(vshr_n_u8(v_src.val[bidx], 3));
                    v_dst = vorrq_u16(v_dst, vshlq_n_u16(vmovl_u8(vand_u8(v_src.val[1], v_n3)), 3));
                    v_dst = vorrq_u16(v_dst, vshlq_n_u16(vmovl_u8(vand_u8(v_src.val[bidx^2], v_n7)), 8));
                    vst1q_u16((ushort *)dst + i, v_dst);
                }
                #endif
                for ( ; i < n; i++, src += 4 )
                    ((ushort*)dst)[i] = (ushort)((src[bidx] >> 3)|((src[1]&~3) << 3)|((src[bidx^2]&~7) << 8));
            }
        }
        else if (scn == 3)
        {
            #if CV_NEON
            for ( ; i <= n - 8; i += 8, src += 24 )
            {
                uint8x8x3_t v_src = vld3_u8(src);
                uint16x8_t v_dst = vmovl_u8(vshr_n_u8(v_src.val[bidx], 3));
                v_dst = vorrq_u16(v_dst, vshlq_n_u16(vmovl_u8(vand_u8(v_src.val[1], v_n7)), 2));
                v_dst = vorrq_u16(v_dst, vshlq_n_u16(vmovl_u8(vand_u8(v_src.val[bidx^2], v_n7)), 7));
                vst1q_u16((ushort *)dst + i, v_dst);
            }
            #endif
            for ( ; i < n; i++, src += 3 )
                ((ushort*)dst)[i] = (ushort)((src[bidx] >> 3)|((src[1]&~7) << 2)|((src[bidx^2]&~7) << 7));
        }
        else
        {
            #if CV_NEON
            for ( ; i <= n - 8; i += 8, src += 32 )
            {
                uint8x8x4_t v_src = vld4_u8(src);
                uint16x8_t v_dst = vmovl_u8(vshr_n_u8(v_src.val[bidx], 3));
                v_dst = vorrq_u16(v_dst, vshlq_n_u16(vmovl_u8(vand_u8(v_src.val[1], v_n7)), 2));
                v_dst = vorrq_u16(v_dst, vorrq_u16(vshlq_n_u16(vmovl_u8(vand_u8(v_src.val[bidx^2], v_n7)), 7),
                                                   vbslq_u16(veorq_u16(vceqq_u16(vmovl_u8(v_src.val[3]), v_0), v_full), v_mask, v_0)));
                vst1q_u16((ushort *)dst + i, v_dst);
            }
            #endif
            for ( ; i < n; i++, src += 4 )
                ((ushort*)dst)[i] = (ushort)((src[bidx] >> 3)|((src[1]&~7) << 2)|
                    ((src[bidx^2]&~7) << 7)|(src[3] ? 0x8000 : 0));
        }
    }

    int srccn, blueIdx, greenBits;
    #if CV_NEON
    uint8x8_t v_n3, v_n7;
    uint16x8_t v_mask, v_0, v_full;
    #endif
};

///////////////////////////////// Color to/from Grayscale ////////////////////////////////

template<typename _Tp>
struct Gray2RGB
{
    typedef _Tp channel_type;

    Gray2RGB(int _dstcn) : dstcn(_dstcn) {}
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        if( dstcn == 3 )
            for( int i = 0; i < n; i++, dst += 3 )
            {
                dst[0] = dst[1] = dst[2] = src[i];
            }
        else
        {
            _Tp alpha = ColorChannel<_Tp>::max();
            for( int i = 0; i < n; i++, dst += 4 )
            {
                dst[0] = dst[1] = dst[2] = src[i];
                dst[3] = alpha;
            }
        }
    }

    int dstcn;
};


struct Gray2RGB5x5
{
    typedef uchar channel_type;

    Gray2RGB5x5(int _greenBits) : greenBits(_greenBits)
    {
        #if CV_NEON
        v_n7 = vdup_n_u8(~7);
        v_n3 = vdup_n_u8(~3);
        #endif
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i = 0;
        if( greenBits == 6 )
        {
            #if CV_NEON
            for ( ; i <= n - 8; i += 8 )
            {
                uint8x8_t v_src = vld1_u8(src + i);
                uint16x8_t v_dst = vmovl_u8(vshr_n_u8(v_src, 3));
                v_dst = vorrq_u16(v_dst, vshlq_n_u16(vmovl_u8(vand_u8(v_src, v_n3)), 3));
                v_dst = vorrq_u16(v_dst, vshlq_n_u16(vmovl_u8(vand_u8(v_src, v_n7)), 8));
                vst1q_u16((ushort *)dst + i, v_dst);
            }
            #endif
            for ( ; i < n; i++ )
            {
                int t = src[i];
                ((ushort*)dst)[i] = (ushort)((t >> 3)|((t & ~3) << 3)|((t & ~7) << 8));
            }
        }
        else
        {
            #if CV_NEON
            for ( ; i <= n - 8; i += 8 )
            {
                uint16x8_t v_src = vmovl_u8(vshr_n_u8(vld1_u8(src + i), 3));
                uint16x8_t v_dst = vorrq_u16(vorrq_u16(v_src, vshlq_n_u16(v_src, 5)), vshlq_n_u16(v_src, 10));
                vst1q_u16((ushort *)dst + i, v_dst);
            }
            #endif
            for( ; i < n; i++ )
            {
                int t = src[i] >> 3;
                ((ushort*)dst)[i] = (ushort)(t|(t << 5)|(t << 10));
            }
        }
    }
    int greenBits;

    #if CV_NEON
    uint8x8_t v_n7, v_n3;
    #endif
};


#undef R2Y
#undef G2Y
#undef B2Y

enum
{
    yuv_shift = 14,
    xyz_shift = 12,
    R2Y = 4899, // == R2YF*16384
    G2Y = 9617, // == G2YF*16384
    B2Y = 1868, // == B2YF*16384
    BLOCK_SIZE = 256
};


struct RGB5x52Gray
{
    typedef uchar channel_type;

    RGB5x52Gray(int _greenBits) : greenBits(_greenBits)
    {
        #if CV_NEON
        v_b2y = vdup_n_u16(B2Y);
        v_g2y = vdup_n_u16(G2Y);
        v_r2y = vdup_n_u16(R2Y);
        v_delta = vdupq_n_u32(1 << (yuv_shift - 1));
        v_f8 = vdupq_n_u16(0xf8);
        v_fc = vdupq_n_u16(0xfc);
        #endif
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i = 0;
        if( greenBits == 6 )
        {
            #if CV_NEON
            for ( ; i <= n - 8; i += 8)
            {
                uint16x8_t v_src = vld1q_u16((ushort *)src + i);
                uint16x8_t v_t0 = vandq_u16(vshlq_n_u16(v_src, 3), v_f8),
                           v_t1 = vandq_u16(vshrq_n_u16(v_src, 3), v_fc),
                           v_t2 = vandq_u16(vshrq_n_u16(v_src, 8), v_f8);

                uint32x4_t v_dst0 = vmlal_u16(vmlal_u16(vmull_u16(vget_low_u16(v_t0), v_b2y),
                                              vget_low_u16(v_t1), v_g2y), vget_low_u16(v_t2), v_r2y);
                uint32x4_t v_dst1 = vmlal_u16(vmlal_u16(vmull_u16(vget_high_u16(v_t0), v_b2y),
                                              vget_high_u16(v_t1), v_g2y), vget_high_u16(v_t2), v_r2y);
                v_dst0 = vshrq_n_u32(vaddq_u32(v_dst0, v_delta), yuv_shift);
                v_dst1 = vshrq_n_u32(vaddq_u32(v_dst1, v_delta), yuv_shift);

                vst1_u8(dst + i, vmovn_u16(vcombine_u16(vmovn_u32(v_dst0), vmovn_u32(v_dst1))));
            }
            #endif
            for ( ; i < n; i++)
            {
                int t = ((ushort*)src)[i];
                dst[i] = (uchar)CV_DESCALE(((t << 3) & 0xf8)*B2Y +
                                           ((t >> 3) & 0xfc)*G2Y +
                                           ((t >> 8) & 0xf8)*R2Y, yuv_shift);
            }
        }
        else
        {
            #if CV_NEON
            for ( ; i <= n - 8; i += 8)
            {
                uint16x8_t v_src = vld1q_u16((ushort *)src + i);
                uint16x8_t v_t0 = vandq_u16(vshlq_n_u16(v_src, 3), v_f8),
                           v_t1 = vandq_u16(vshrq_n_u16(v_src, 2), v_f8),
                           v_t2 = vandq_u16(vshrq_n_u16(v_src, 7), v_f8);

                uint32x4_t v_dst0 = vmlal_u16(vmlal_u16(vmull_u16(vget_low_u16(v_t0), v_b2y),
                                              vget_low_u16(v_t1), v_g2y), vget_low_u16(v_t2), v_r2y);
                uint32x4_t v_dst1 = vmlal_u16(vmlal_u16(vmull_u16(vget_high_u16(v_t0), v_b2y),
                                              vget_high_u16(v_t1), v_g2y), vget_high_u16(v_t2), v_r2y);
                v_dst0 = vshrq_n_u32(vaddq_u32(v_dst0, v_delta), yuv_shift);
                v_dst1 = vshrq_n_u32(vaddq_u32(v_dst1, v_delta), yuv_shift);

                vst1_u8(dst + i, vmovn_u16(vcombine_u16(vmovn_u32(v_dst0), vmovn_u32(v_dst1))));
            }
            #endif
            for ( ; i < n; i++)
            {
                int t = ((ushort*)src)[i];
                dst[i] = (uchar)CV_DESCALE(((t << 3) & 0xf8)*B2Y +
                                           ((t >> 2) & 0xf8)*G2Y +
                                           ((t >> 7) & 0xf8)*R2Y, yuv_shift);
            }
        }
    }
    int greenBits;

    #if CV_NEON
    uint16x4_t v_b2y, v_g2y, v_r2y;
    uint32x4_t v_delta;
    uint16x8_t v_f8, v_fc;
    #endif
};


template<typename _Tp> struct RGB2Gray
{
    typedef _Tp channel_type;

    RGB2Gray(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        static const float coeffs0[] = { R2YF, G2YF, B2YF };
        memcpy( coeffs, _coeffs ? _coeffs : coeffs0, 3*sizeof(coeffs[0]) );
        if(blueIdx == 0)
            std::swap(coeffs[0], coeffs[2]);
    }

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn;
        float cb = coeffs[0], cg = coeffs[1], cr = coeffs[2];
        for(int i = 0; i < n; i++, src += scn)
            dst[i] = saturate_cast<_Tp>(src[0]*cb + src[1]*cg + src[2]*cr);
    }
    int srccn;
    float coeffs[3];
};

template<> struct RGB2Gray<uchar>
{
    typedef uchar channel_type;

    RGB2Gray(int _srccn, int blueIdx, const int* coeffs) : srccn(_srccn)
    {
        const int coeffs0[] = { R2Y, G2Y, B2Y };
        if(!coeffs) coeffs = coeffs0;

        int b = 0, g = 0, r = (1 << (yuv_shift-1));
        int db = coeffs[blueIdx^2], dg = coeffs[1], dr = coeffs[blueIdx];

        for( int i = 0; i < 256; i++, b += db, g += dg, r += dr )
        {
            tab[i] = b;
            tab[i+256] = g;
            tab[i+512] = r;
        }
    }
    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int scn = srccn;
        const int* _tab = tab;
        for(int i = 0; i < n; i++, src += scn)
            dst[i] = (uchar)((_tab[src[0]] + _tab[src[1]+256] + _tab[src[2]+512]) >> yuv_shift);
    }
    int srccn;
    int tab[256*3];
};

#if CV_NEON

template <>
struct RGB2Gray<ushort>
{
    typedef ushort channel_type;

    RGB2Gray(int _srccn, int blueIdx, const int* _coeffs) :
        srccn(_srccn)
    {
        static const int coeffs0[] = { R2Y, G2Y, B2Y };
        memcpy(coeffs, _coeffs ? _coeffs : coeffs0, 3*sizeof(coeffs[0]));
        if( blueIdx == 0 )
            std::swap(coeffs[0], coeffs[2]);

        v_cb = vdup_n_u16(coeffs[0]);
        v_cg = vdup_n_u16(coeffs[1]);
        v_cr = vdup_n_u16(coeffs[2]);
        v_delta = vdupq_n_u32(1 << (yuv_shift - 1));
    }

    void operator()(const ushort* src, ushort* dst, int n) const
    {
        int scn = srccn, cb = coeffs[0], cg = coeffs[1], cr = coeffs[2], i = 0;

        for ( ; i <= n - 8; i += 8, src += scn * 8)
        {
            uint16x8_t v_b, v_r, v_g;
            if (scn == 3)
            {
                uint16x8x3_t v_src = vld3q_u16(src);
                v_b = v_src.val[0];
                v_g = v_src.val[1];
                v_r = v_src.val[2];
            }
            else
            {
                uint16x8x4_t v_src = vld4q_u16(src);
                v_b = v_src.val[0];
                v_g = v_src.val[1];
                v_r = v_src.val[2];
            }

            uint32x4_t v_dst0_ = vmlal_u16(vmlal_u16(
                                           vmull_u16(vget_low_u16(v_b), v_cb),
                                                     vget_low_u16(v_g), v_cg),
                                                     vget_low_u16(v_r), v_cr);
            uint32x4_t v_dst1_ = vmlal_u16(vmlal_u16(
                                           vmull_u16(vget_high_u16(v_b), v_cb),
                                                     vget_high_u16(v_g), v_cg),
                                                     vget_high_u16(v_r), v_cr);

            uint16x4_t v_dst0 = vmovn_u32(vshrq_n_u32(vaddq_u32(v_dst0_, v_delta), yuv_shift));
            uint16x4_t v_dst1 = vmovn_u32(vshrq_n_u32(vaddq_u32(v_dst1_, v_delta), yuv_shift));

            vst1q_u16(dst + i, vcombine_u16(v_dst0, v_dst1));
        }

        for ( ; i <= n - 4; i += 4, src += scn * 4)
        {
            uint16x4_t v_b, v_r, v_g;
            if (scn == 3)
            {
                uint16x4x3_t v_src = vld3_u16(src);
                v_b = v_src.val[0];
                v_g = v_src.val[1];
                v_r = v_src.val[2];
            }
            else
            {
                uint16x4x4_t v_src = vld4_u16(src);
                v_b = v_src.val[0];
                v_g = v_src.val[1];
                v_r = v_src.val[2];
            }

            uint32x4_t v_dst = vmlal_u16(vmlal_u16(
                                         vmull_u16(v_b, v_cb),
                                                   v_g, v_cg),
                                                   v_r, v_cr);

            vst1_u16(dst + i, vmovn_u32(vshrq_n_u32(vaddq_u32(v_dst, v_delta), yuv_shift)));
        }

        for( ; i < n; i++, src += scn)
            dst[i] = (ushort)CV_DESCALE((unsigned)(src[0]*cb + src[1]*cg + src[2]*cr), yuv_shift);
    }

    int srccn, coeffs[3];
    uint16x4_t v_cb, v_cg, v_cr;
    uint32x4_t v_delta;
};

template <>
struct RGB2Gray<float>
{
    typedef float channel_type;

    RGB2Gray(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        static const float coeffs0[] = { R2YF, G2YF, B2YF };
        memcpy( coeffs, _coeffs ? _coeffs : coeffs0, 3*sizeof(coeffs[0]) );
        if(blueIdx == 0)
            std::swap(coeffs[0], coeffs[2]);

        v_cb = vdupq_n_f32(coeffs[0]);
        v_cg = vdupq_n_f32(coeffs[1]);
        v_cr = vdupq_n_f32(coeffs[2]);
    }

    void operator()(const float * src, float * dst, int n) const
    {
        int scn = srccn, i = 0;
        float cb = coeffs[0], cg = coeffs[1], cr = coeffs[2];

        if (scn == 3)
        {
            for ( ; i <= n - 8; i += 8, src += scn * 8)
            {
                float32x4x3_t v_src = vld3q_f32(src);
                vst1q_f32(dst + i, vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_cb), v_src.val[1], v_cg), v_src.val[2], v_cr));

                v_src = vld3q_f32(src + scn * 4);
                vst1q_f32(dst + i + 4, vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_cb), v_src.val[1], v_cg), v_src.val[2], v_cr));
            }

            for ( ; i <= n - 4; i += 4, src += scn * 4)
            {
                float32x4x3_t v_src = vld3q_f32(src);
                vst1q_f32(dst + i, vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_cb), v_src.val[1], v_cg), v_src.val[2], v_cr));
            }
        }
        else
        {
            for ( ; i <= n - 8; i += 8, src += scn * 8)
            {
                float32x4x4_t v_src = vld4q_f32(src);
                vst1q_f32(dst + i, vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_cb), v_src.val[1], v_cg), v_src.val[2], v_cr));

                v_src = vld4q_f32(src + scn * 4);
                vst1q_f32(dst + i + 4, vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_cb), v_src.val[1], v_cg), v_src.val[2], v_cr));
            }

            for ( ; i <= n - 4; i += 4, src += scn * 4)
            {
                float32x4x4_t v_src = vld4q_f32(src);
                vst1q_f32(dst + i, vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_cb), v_src.val[1], v_cg), v_src.val[2], v_cr));
            }
        }

        for ( ; i < n; i++, src += scn)
            dst[i] = src[0]*cb + src[1]*cg + src[2]*cr;
    }

    int srccn;
    float coeffs[3];
    float32x4_t v_cb, v_cg, v_cr;
};

#endif // CV_SSE2

#if !CV_NEON && !CV_SSE4_1

template<> struct RGB2Gray<ushort>
{
    typedef ushort channel_type;

    RGB2Gray(int _srccn, int blueIdx, const int* _coeffs) : srccn(_srccn)
    {
        static const int coeffs0[] = { R2Y, G2Y, B2Y };
        memcpy(coeffs, _coeffs ? _coeffs : coeffs0, 3*sizeof(coeffs[0]));
        if( blueIdx == 0 )
            std::swap(coeffs[0], coeffs[2]);
    }

    void operator()(const ushort* src, ushort* dst, int n) const
    {
        int scn = srccn, cb = coeffs[0], cg = coeffs[1], cr = coeffs[2];
        for(int i = 0; i < n; i++, src += scn)
            dst[i] = (ushort)CV_DESCALE((unsigned)(src[0]*cb + src[1]*cg + src[2]*cr), yuv_shift);
    }
    int srccn;
    int coeffs[3];
};

#endif // !CV_NEON && !CV_SSE4_1

///////////////////////////////////// RGB <-> YCrCb //////////////////////////////////////

template<typename _Tp> struct RGB2YCrCb_f
{
    typedef _Tp channel_type;

    RGB2YCrCb_f(int _srccn, int _blueIdx, bool _isCrCb) : srccn(_srccn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const float coeffs_crb[] = { R2YF, G2YF, B2YF, YCRF, YCBF };
        static const float coeffs_yuv[] = { R2YF, G2YF, B2YF, R2VF, B2UF };
        memcpy(coeffs, isCrCb ? coeffs_crb : coeffs_yuv, 5*sizeof(coeffs[0]));
        if(blueIdx==0) std::swap(coeffs[0], coeffs[2]);
    }

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn, bidx = blueIdx;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        const _Tp delta = ColorChannel<_Tp>::half();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3], C4 = coeffs[4];
        n *= 3;
        for(int i = 0; i < n; i += 3, src += scn)
        {
            _Tp Y = saturate_cast<_Tp>(src[0]*C0 + src[1]*C1 + src[2]*C2);
            _Tp Cr = saturate_cast<_Tp>((src[bidx^2] - Y)*C3 + delta);
            _Tp Cb = saturate_cast<_Tp>((src[bidx] - Y)*C4 + delta);
            dst[i] = Y; dst[i+1+yuvOrder] = Cr; dst[i+2-yuvOrder] = Cb;
        }
    }
    int srccn, blueIdx;
    bool isCrCb;
    float coeffs[5];
};

#if CV_NEON

template <>
struct RGB2YCrCb_f<float>
{
    typedef float channel_type;

    RGB2YCrCb_f(int _srccn, int _blueIdx, bool _isCrCb) :
        srccn(_srccn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const float coeffs_crb[] = { R2YF, G2YF, B2YF, YCRF, YCBF };
        static const float coeffs_yuv[] = { R2YF, G2YF, B2YF, R2VF, B2UF };
        memcpy(coeffs, isCrCb ? coeffs_crb : coeffs_yuv, 5*sizeof(coeffs[0]));
        if(blueIdx==0)
            std::swap(coeffs[0], coeffs[2]);

        v_c0 = vdupq_n_f32(coeffs[0]);
        v_c1 = vdupq_n_f32(coeffs[1]);
        v_c2 = vdupq_n_f32(coeffs[2]);
        v_c3 = vdupq_n_f32(coeffs[3]);
        v_c4 = vdupq_n_f32(coeffs[4]);
        v_delta = vdupq_n_f32(ColorChannel<float>::half());
    }

    void operator()(const float * src, float * dst, int n) const
    {
        int scn = srccn, bidx = blueIdx, i = 0;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        const float delta = ColorChannel<float>::half();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3], C4 = coeffs[4];
        n *= 3;

        if (scn == 3)
            for ( ; i <= n - 12; i += 12, src += 12)
            {
                float32x4x3_t v_src = vld3q_f32(src), v_dst;
                v_dst.val[0] = vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_c0), v_src.val[1], v_c1), v_src.val[2], v_c2);
                v_dst.val[1+yuvOrder] = vmlaq_f32(v_delta, vsubq_f32(v_src.val[bidx^2], v_dst.val[0]), v_c3);
                v_dst.val[2-yuvOrder] = vmlaq_f32(v_delta, vsubq_f32(v_src.val[bidx], v_dst.val[0]), v_c4);

                vst3q_f32(dst + i, v_dst);
            }
        else
            for ( ; i <= n - 12; i += 12, src += 16)
            {
                float32x4x4_t v_src = vld4q_f32(src);
                float32x4x3_t v_dst;
                v_dst.val[0] = vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_c0), v_src.val[1], v_c1), v_src.val[2], v_c2);
                v_dst.val[1+yuvOrder] = vmlaq_f32(v_delta, vsubq_f32(v_src.val[bidx^2], v_dst.val[0]), v_c3);
                v_dst.val[2-yuvOrder] = vmlaq_f32(v_delta, vsubq_f32(v_src.val[bidx], v_dst.val[0]), v_c4);

                vst3q_f32(dst + i, v_dst);
            }

        for ( ; i < n; i += 3, src += scn)
        {
            float Y = src[0]*C0 + src[1]*C1 + src[2]*C2;
            float Cr = (src[bidx^2] - Y)*C3 + delta;
            float Cb = (src[bidx] - Y)*C4 + delta;
            dst[i] = Y; dst[i+1+yuvOrder] = Cr; dst[i+2-yuvOrder] = Cb;
        }
    }
    int srccn, blueIdx;
    bool isCrCb;
    float coeffs[5];
    float32x4_t v_c0, v_c1, v_c2, v_c3, v_c4, v_delta;
};
#endif

template<typename _Tp> struct RGB2YCrCb_i
{
    typedef _Tp channel_type;

    RGB2YCrCb_i(int _srccn, int _blueIdx, bool _isCrCb)
        : srccn(_srccn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const int coeffs_crb[] = { R2Y, G2Y, B2Y, YCRI, YCBI };
        static const int coeffs_yuv[] = { R2Y, G2Y, B2Y, R2VI, B2UI };
        memcpy(coeffs, isCrCb ? coeffs_crb : coeffs_yuv, 5*sizeof(coeffs[0]));
        if(blueIdx==0) std::swap(coeffs[0], coeffs[2]);
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn, bidx = blueIdx;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3], C4 = coeffs[4];
        int delta = ColorChannel<_Tp>::half()*(1 << yuv_shift);
        n *= 3;
        for(int i = 0; i < n; i += 3, src += scn)
        {
            int Y = CV_DESCALE(src[0]*C0 + src[1]*C1 + src[2]*C2, yuv_shift);
            int Cr = CV_DESCALE((src[bidx^2] - Y)*C3 + delta, yuv_shift);
            int Cb = CV_DESCALE((src[bidx] - Y)*C4 + delta, yuv_shift);
            dst[i] = saturate_cast<_Tp>(Y);
            dst[i+1+yuvOrder] = saturate_cast<_Tp>(Cr);
            dst[i+2-yuvOrder] = saturate_cast<_Tp>(Cb);
        }
    }
    int srccn, blueIdx;
    bool isCrCb;
    int coeffs[5];
};

#if CV_NEON

template <>
struct RGB2YCrCb_i<uchar>
{
    typedef uchar channel_type;

    RGB2YCrCb_i(int _srccn, int _blueIdx, bool _isCrCb)
        : srccn(_srccn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const int coeffs_crb[] = { R2Y, G2Y, B2Y, YCRI, YCBI };
        static const int coeffs_yuv[] = { R2Y, G2Y, B2Y, R2VI, B2UI };
        memcpy(coeffs, isCrCb ? coeffs_crb : coeffs_yuv, 5*sizeof(coeffs[0]));
        if (blueIdx==0)
            std::swap(coeffs[0], coeffs[2]);

        v_c0 = vdup_n_s16(coeffs[0]);
        v_c1 = vdup_n_s16(coeffs[1]);
        v_c2 = vdup_n_s16(coeffs[2]);
        v_c3 = vdupq_n_s32(coeffs[3]);
        v_c4 = vdupq_n_s32(coeffs[4]);
        v_delta = vdupq_n_s32(ColorChannel<uchar>::half()*(1 << yuv_shift));
        v_delta2 = vdupq_n_s32(1 << (yuv_shift - 1));
    }

    void operator()(const uchar * src, uchar * dst, int n) const
    {
        int scn = srccn, bidx = blueIdx, i = 0;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3], C4 = coeffs[4];
        int delta = ColorChannel<uchar>::half()*(1 << yuv_shift);
        n *= 3;

        for ( ; i <= n - 24; i += 24, src += scn * 8)
        {
            uint8x8x3_t v_dst;
            int16x8x3_t v_src16;

            if (scn == 3)
            {
                uint8x8x3_t v_src = vld3_u8(src);
                v_src16.val[0] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[0]));
                v_src16.val[1] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[1]));
                v_src16.val[2] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[2]));
            }
            else
            {
                uint8x8x4_t v_src = vld4_u8(src);
                v_src16.val[0] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[0]));
                v_src16.val[1] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[1]));
                v_src16.val[2] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[2]));
            }

            int16x4x3_t v_src0;
            v_src0.val[0] = vget_low_s16(v_src16.val[0]);
            v_src0.val[1] = vget_low_s16(v_src16.val[1]);
            v_src0.val[2] = vget_low_s16(v_src16.val[2]);

            int32x4_t v_Y0 = vmlal_s16(vmlal_s16(vmull_s16(v_src0.val[0], v_c0), v_src0.val[1], v_c1), v_src0.val[2], v_c2);
            v_Y0 = vshrq_n_s32(vaddq_s32(v_Y0, v_delta2), yuv_shift);
            int32x4_t v_Cr0 = vmlaq_s32(v_delta, vsubq_s32(vmovl_s16(v_src0.val[bidx^2]), v_Y0), v_c3);
            v_Cr0 = vshrq_n_s32(vaddq_s32(v_Cr0, v_delta2), yuv_shift);
            int32x4_t v_Cb0 = vmlaq_s32(v_delta, vsubq_s32(vmovl_s16(v_src0.val[bidx]), v_Y0), v_c4);
            v_Cb0 = vshrq_n_s32(vaddq_s32(v_Cb0, v_delta2), yuv_shift);

            v_src0.val[0] = vget_high_s16(v_src16.val[0]);
            v_src0.val[1] = vget_high_s16(v_src16.val[1]);
            v_src0.val[2] = vget_high_s16(v_src16.val[2]);

            int32x4_t v_Y1 = vmlal_s16(vmlal_s16(vmull_s16(v_src0.val[0], v_c0), v_src0.val[1], v_c1), v_src0.val[2], v_c2);
            v_Y1 = vshrq_n_s32(vaddq_s32(v_Y1, v_delta2), yuv_shift);
            int32x4_t v_Cr1 = vmlaq_s32(v_delta, vsubq_s32(vmovl_s16(v_src0.val[bidx^2]), v_Y1), v_c3);
            v_Cr1 = vshrq_n_s32(vaddq_s32(v_Cr1, v_delta2), yuv_shift);
            int32x4_t v_Cb1 = vmlaq_s32(v_delta, vsubq_s32(vmovl_s16(v_src0.val[bidx]), v_Y1), v_c4);
            v_Cb1 = vshrq_n_s32(vaddq_s32(v_Cb1, v_delta2), yuv_shift);

            v_dst.val[0] = vqmovun_s16(vcombine_s16(vqmovn_s32(v_Y0), vqmovn_s32(v_Y1)));
            v_dst.val[1+yuvOrder] = vqmovun_s16(vcombine_s16(vqmovn_s32(v_Cr0), vqmovn_s32(v_Cr1)));
            v_dst.val[2-yuvOrder] = vqmovun_s16(vcombine_s16(vqmovn_s32(v_Cb0), vqmovn_s32(v_Cb1)));

            vst3_u8(dst + i, v_dst);
        }

        for ( ; i < n; i += 3, src += scn)
        {
            int Y = CV_DESCALE(src[0]*C0 + src[1]*C1 + src[2]*C2, yuv_shift);
            int Cr = CV_DESCALE((src[bidx^2] - Y)*C3 + delta, yuv_shift);
            int Cb = CV_DESCALE((src[bidx] - Y)*C4 + delta, yuv_shift);
            dst[i] = saturate_cast<uchar>(Y);
            dst[i+1+yuvOrder] = saturate_cast<uchar>(Cr);
            dst[i+2-yuvOrder] = saturate_cast<uchar>(Cb);
        }
    }
    int srccn, blueIdx, coeffs[5];
    bool isCrCb;
    int16x4_t v_c0, v_c1, v_c2;
    int32x4_t v_c3, v_c4, v_delta, v_delta2;
};

template <>
struct RGB2YCrCb_i<ushort>
{
    typedef ushort channel_type;

    RGB2YCrCb_i(int _srccn, int _blueIdx, bool _isCrCb)
        : srccn(_srccn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const int coeffs_crb[] = { R2Y, G2Y, B2Y, YCRI, YCBI };
        static const int coeffs_yuv[] = { R2Y, G2Y, B2Y, R2VI, B2UI };
        memcpy(coeffs, isCrCb ? coeffs_crb : coeffs_yuv, 5*sizeof(coeffs[0]));
        if (blueIdx==0)
            std::swap(coeffs[0], coeffs[2]);

        v_c0 = vdupq_n_s32(coeffs[0]);
        v_c1 = vdupq_n_s32(coeffs[1]);
        v_c2 = vdupq_n_s32(coeffs[2]);
        v_c3 = vdupq_n_s32(coeffs[3]);
        v_c4 = vdupq_n_s32(coeffs[4]);
        v_delta = vdupq_n_s32(ColorChannel<ushort>::half()*(1 << yuv_shift));
        v_delta2 = vdupq_n_s32(1 << (yuv_shift - 1));
    }

    void operator()(const ushort * src, ushort * dst, int n) const
    {
        int scn = srccn, bidx = blueIdx, i = 0;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3], C4 = coeffs[4];
        int delta = ColorChannel<ushort>::half()*(1 << yuv_shift);
        n *= 3;

        for ( ; i <= n - 24; i += 24, src += scn * 8)
        {
            uint16x8x3_t v_src, v_dst;
            int32x4x3_t v_src0;

            if (scn == 3)
                v_src = vld3q_u16(src);
            else
            {
                uint16x8x4_t v_src_ = vld4q_u16(src);
                v_src.val[0] = v_src_.val[0];
                v_src.val[1] = v_src_.val[1];
                v_src.val[2] = v_src_.val[2];
            }

            v_src0.val[0] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src.val[0])));
            v_src0.val[1] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src.val[1])));
            v_src0.val[2] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src.val[2])));

            int32x4_t v_Y0 = vmlaq_s32(vmlaq_s32(vmulq_s32(v_src0.val[0], v_c0), v_src0.val[1], v_c1), v_src0.val[2], v_c2);
            v_Y0 = vshrq_n_s32(vaddq_s32(v_Y0, v_delta2), yuv_shift);
            int32x4_t v_Cr0 = vmlaq_s32(v_delta, vsubq_s32(v_src0.val[bidx^2], v_Y0), v_c3);
            v_Cr0 = vshrq_n_s32(vaddq_s32(v_Cr0, v_delta2), yuv_shift);
            int32x4_t v_Cb0 = vmlaq_s32(v_delta, vsubq_s32(v_src0.val[bidx], v_Y0), v_c4);
            v_Cb0 = vshrq_n_s32(vaddq_s32(v_Cb0, v_delta2), yuv_shift);

            v_src0.val[0] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src.val[0])));
            v_src0.val[1] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src.val[1])));
            v_src0.val[2] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src.val[2])));

            int32x4_t v_Y1 = vmlaq_s32(vmlaq_s32(vmulq_s32(v_src0.val[0], v_c0), v_src0.val[1], v_c1), v_src0.val[2], v_c2);
            v_Y1 = vshrq_n_s32(vaddq_s32(v_Y1, v_delta2), yuv_shift);
            int32x4_t v_Cr1 = vmlaq_s32(v_delta, vsubq_s32(v_src0.val[bidx^2], v_Y1), v_c3);
            v_Cr1 = vshrq_n_s32(vaddq_s32(v_Cr1, v_delta2), yuv_shift);
            int32x4_t v_Cb1 = vmlaq_s32(v_delta, vsubq_s32(v_src0.val[bidx], v_Y1), v_c4);
            v_Cb1 = vshrq_n_s32(vaddq_s32(v_Cb1, v_delta2), yuv_shift);

            v_dst.val[0] = vcombine_u16(vqmovun_s32(v_Y0), vqmovun_s32(v_Y1));
            v_dst.val[1+yuvOrder] = vcombine_u16(vqmovun_s32(v_Cr0), vqmovun_s32(v_Cr1));
            v_dst.val[2-yuvOrder] = vcombine_u16(vqmovun_s32(v_Cb0), vqmovun_s32(v_Cb1));

            vst3q_u16(dst + i, v_dst);
        }

        for ( ; i <= n - 12; i += 12, src += scn * 4)
        {
            uint16x4x3_t v_dst;
            int32x4x3_t v_src0;

            if (scn == 3)
            {
                uint16x4x3_t v_src = vld3_u16(src);
                v_src0.val[0] = vreinterpretq_s32_u32(vmovl_u16(v_src.val[0]));
                v_src0.val[1] = vreinterpretq_s32_u32(vmovl_u16(v_src.val[1]));
                v_src0.val[2] = vreinterpretq_s32_u32(vmovl_u16(v_src.val[2]));
            }
            else
            {
                uint16x4x4_t v_src = vld4_u16(src);
                v_src0.val[0] = vreinterpretq_s32_u32(vmovl_u16(v_src.val[0]));
                v_src0.val[1] = vreinterpretq_s32_u32(vmovl_u16(v_src.val[1]));
                v_src0.val[2] = vreinterpretq_s32_u32(vmovl_u16(v_src.val[2]));
            }

            int32x4_t v_Y = vmlaq_s32(vmlaq_s32(vmulq_s32(v_src0.val[0], v_c0), v_src0.val[1], v_c1), v_src0.val[2], v_c2);
            v_Y = vshrq_n_s32(vaddq_s32(v_Y, v_delta2), yuv_shift);
            int32x4_t v_Cr = vmlaq_s32(v_delta, vsubq_s32(v_src0.val[bidx^2], v_Y), v_c3);
            v_Cr = vshrq_n_s32(vaddq_s32(v_Cr, v_delta2), yuv_shift);
            int32x4_t v_Cb = vmlaq_s32(v_delta, vsubq_s32(v_src0.val[bidx], v_Y), v_c4);
            v_Cb = vshrq_n_s32(vaddq_s32(v_Cb, v_delta2), yuv_shift);

            v_dst.val[0] = vqmovun_s32(v_Y);
            v_dst.val[1+yuvOrder] = vqmovun_s32(v_Cr);
            v_dst.val[2-yuvOrder] = vqmovun_s32(v_Cb);

            vst3_u16(dst + i, v_dst);
        }

        for ( ; i < n; i += 3, src += scn)
        {
            int Y = CV_DESCALE(src[0]*C0 + src[1]*C1 + src[2]*C2, yuv_shift);
            int Cr = CV_DESCALE((src[bidx^2] - Y)*C3 + delta, yuv_shift);
            int Cb = CV_DESCALE((src[bidx] - Y)*C4 + delta, yuv_shift);
            dst[i] = saturate_cast<ushort>(Y);
            dst[i+1+yuvOrder] = saturate_cast<ushort>(Cr);
            dst[i+2-yuvOrder] = saturate_cast<ushort>(Cb);
        }
    }
    int srccn, blueIdx, coeffs[5];
    bool isCrCb;
    int32x4_t v_c0, v_c1, v_c2, v_c3, v_c4, v_delta, v_delta2;
};
#endif // CV_SSE4_1

template<typename _Tp> struct YCrCb2RGB_f
{
    typedef _Tp channel_type;

    YCrCb2RGB_f(int _dstcn, int _blueIdx, bool _isCrCb)
        : dstcn(_dstcn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const float coeffs_cbr[] = {CR2RF, CR2GF, CB2GF, CB2BF};
        static const float coeffs_yuv[] = { V2RF,  V2GF,  U2GF,  U2BF};
        memcpy(coeffs, isCrCb ? coeffs_cbr : coeffs_yuv, 4*sizeof(coeffs[0]));
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        const _Tp delta = ColorChannel<_Tp>::half(), alpha = ColorChannel<_Tp>::max();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3];
        n *= 3;
        for(int i = 0; i < n; i += 3, dst += dcn)
        {
            _Tp Y = src[i];
            _Tp Cr = src[i+1+yuvOrder];
            _Tp Cb = src[i+2-yuvOrder];

            _Tp b = saturate_cast<_Tp>(Y + (Cb - delta)*C3);
            _Tp g = saturate_cast<_Tp>(Y + (Cb - delta)*C2 + (Cr - delta)*C1);
            _Tp r = saturate_cast<_Tp>(Y + (Cr - delta)*C0);

            dst[bidx] = b; dst[1] = g; dst[bidx^2] = r;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    bool isCrCb;
    float coeffs[4];
};

#if CV_NEON

template <>
struct YCrCb2RGB_f<float>
{
    typedef float channel_type;

    YCrCb2RGB_f(int _dstcn, int _blueIdx, bool _isCrCb)
        : dstcn(_dstcn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const float coeffs_cbr[] = {CR2RF, CR2GF, CB2GF, CB2BF};
        static const float coeffs_yuv[] = { V2RF,  V2GF,  U2GF,  U2BF};
        memcpy(coeffs, isCrCb ? coeffs_cbr : coeffs_yuv, 4*sizeof(coeffs[0]));

        v_c0 = vdupq_n_f32(coeffs[0]);
        v_c1 = vdupq_n_f32(coeffs[1]);
        v_c2 = vdupq_n_f32(coeffs[2]);
        v_c3 = vdupq_n_f32(coeffs[3]);
        v_delta = vdupq_n_f32(ColorChannel<float>::half());
        v_alpha = vdupq_n_f32(ColorChannel<float>::max());
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx, i = 0;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        const float delta = ColorChannel<float>::half(), alpha = ColorChannel<float>::max();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3];
        n *= 3;

        if (dcn == 3)
            for ( ; i <= n - 12; i += 12, dst += 12)
            {
                float32x4x3_t v_src = vld3q_f32(src + i), v_dst;
                float32x4_t v_Y = v_src.val[0], v_Cr = v_src.val[1+yuvOrder], v_Cb = v_src.val[2-yuvOrder];

                v_dst.val[bidx] = vmlaq_f32(v_Y, vsubq_f32(v_Cb, v_delta), v_c3);
                v_dst.val[1] = vaddq_f32(vmlaq_f32(vmulq_f32(vsubq_f32(v_Cb, v_delta), v_c2), vsubq_f32(v_Cr, v_delta), v_c1), v_Y);
                v_dst.val[bidx^2] = vmlaq_f32(v_Y, vsubq_f32(v_Cr, v_delta), v_c0);

                vst3q_f32(dst, v_dst);
            }
        else
            for ( ; i <= n - 12; i += 12, dst += 16)
            {
                float32x4x3_t v_src = vld3q_f32(src + i);
                float32x4x4_t v_dst;
                float32x4_t v_Y = v_src.val[0], v_Cr = v_src.val[1+yuvOrder], v_Cb = v_src.val[2-yuvOrder];

                v_dst.val[bidx] = vmlaq_f32(v_Y, vsubq_f32(v_Cb, v_delta), v_c3);
                v_dst.val[1] = vaddq_f32(vmlaq_f32(vmulq_f32(vsubq_f32(v_Cb, v_delta), v_c2), vsubq_f32(v_Cr, v_delta), v_c1), v_Y);
                v_dst.val[bidx^2] = vmlaq_f32(v_Y, vsubq_f32(v_Cr, v_delta), v_c0);
                v_dst.val[3] = v_alpha;

                vst4q_f32(dst, v_dst);
            }

        for ( ; i < n; i += 3, dst += dcn)
        {
            float Y = src[i], Cr = src[i+1+yuvOrder], Cb = src[i+2-yuvOrder];

            float b = Y + (Cb - delta)*C3;
            float g = Y + (Cb - delta)*C2 + (Cr - delta)*C1;
            float r = Y + (Cr - delta)*C0;

            dst[bidx] = b; dst[1] = g; dst[bidx^2] = r;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    bool isCrCb;
    float coeffs[4];
    float32x4_t v_c0, v_c1, v_c2, v_c3, v_alpha, v_delta;
};
#endif

template<typename _Tp> struct YCrCb2RGB_i
{
    typedef _Tp channel_type;

    YCrCb2RGB_i(int _dstcn, int _blueIdx, bool _isCrCb)
        : dstcn(_dstcn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const int coeffs_crb[] = { CR2RI, CR2GI, CB2GI, CB2BI};
        static const int coeffs_yuv[] = {  V2RI,  V2GI,  U2GI, U2BI };
        memcpy(coeffs, isCrCb ? coeffs_crb : coeffs_yuv, 4*sizeof(coeffs[0]));
    }

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        const _Tp delta = ColorChannel<_Tp>::half(), alpha = ColorChannel<_Tp>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3];
        n *= 3;
        for(int i = 0; i < n; i += 3, dst += dcn)
        {
            _Tp Y = src[i];
            _Tp Cr = src[i+1+yuvOrder];
            _Tp Cb = src[i+2-yuvOrder];

            int b = Y + CV_DESCALE((Cb - delta)*C3, yuv_shift);
            int g = Y + CV_DESCALE((Cb - delta)*C2 + (Cr - delta)*C1, yuv_shift);
            int r = Y + CV_DESCALE((Cr - delta)*C0, yuv_shift);

            dst[bidx] = saturate_cast<_Tp>(b);
            dst[1] = saturate_cast<_Tp>(g);
            dst[bidx^2] = saturate_cast<_Tp>(r);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    bool isCrCb;
    int coeffs[4];
};

#if CV_NEON

template <>
struct YCrCb2RGB_i<uchar>
{
    typedef uchar channel_type;

    YCrCb2RGB_i(int _dstcn, int _blueIdx, bool _isCrCb)
        : dstcn(_dstcn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const int coeffs_crb[] = { CR2RI, CR2GI, CB2GI, CB2BI};
        static const int coeffs_yuv[] = {  V2RI,  V2GI,  U2GI, U2BI };
        memcpy(coeffs, isCrCb ? coeffs_crb : coeffs_yuv, 4*sizeof(coeffs[0]));

        v_c0 = vdupq_n_s32(coeffs[0]);
        v_c1 = vdupq_n_s32(coeffs[1]);
        v_c2 = vdupq_n_s32(coeffs[2]);
        v_c3 = vdupq_n_s32(coeffs[3]);
        v_delta = vdup_n_s16(ColorChannel<uchar>::half());
        v_delta2 = vdupq_n_s32(1 << (yuv_shift - 1));
        v_alpha = vdup_n_u8(ColorChannel<uchar>::max());
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx, i = 0;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        const uchar delta = ColorChannel<uchar>::half(), alpha = ColorChannel<uchar>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3];
        n *= 3;

        for ( ; i <= n - 24; i += 24, dst += dcn * 8)
        {
            uint8x8x3_t v_src = vld3_u8(src + i);
            int16x8x3_t v_src16;
            v_src16.val[0] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[0]));
            v_src16.val[1] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[1]));
            v_src16.val[2] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[2]));

            int16x4_t v_Y = vget_low_s16(v_src16.val[0]),
                      v_Cr = vget_low_s16(v_src16.val[1+yuvOrder]),
                      v_Cb = vget_low_s16(v_src16.val[2-yuvOrder]);

            int32x4_t v_b0 = vmulq_s32(v_c3, vsubl_s16(v_Cb, v_delta));
            v_b0 = vaddw_s16(vshrq_n_s32(vaddq_s32(v_b0, v_delta2), yuv_shift), v_Y);
            int32x4_t v_g0 = vmlaq_s32(vmulq_s32(vsubl_s16(v_Cr, v_delta), v_c1), vsubl_s16(v_Cb, v_delta), v_c2);
            v_g0 = vaddw_s16(vshrq_n_s32(vaddq_s32(v_g0, v_delta2), yuv_shift), v_Y);
            int32x4_t v_r0 = vmulq_s32(v_c0, vsubl_s16(v_Cr, v_delta));
            v_r0 = vaddw_s16(vshrq_n_s32(vaddq_s32(v_r0, v_delta2), yuv_shift), v_Y);

            v_Y = vget_high_s16(v_src16.val[0]);
            v_Cr = vget_high_s16(v_src16.val[1+yuvOrder]);
            v_Cb = vget_high_s16(v_src16.val[2-yuvOrder]);

            int32x4_t v_b1 = vmulq_s32(v_c3, vsubl_s16(v_Cb, v_delta));
            v_b1 = vaddw_s16(vshrq_n_s32(vaddq_s32(v_b1, v_delta2), yuv_shift), v_Y);
            int32x4_t v_g1 = vmlaq_s32(vmulq_s32(vsubl_s16(v_Cr, v_delta), v_c1), vsubl_s16(v_Cb, v_delta), v_c2);
            v_g1 = vaddw_s16(vshrq_n_s32(vaddq_s32(v_g1, v_delta2), yuv_shift), v_Y);
            int32x4_t v_r1 = vmulq_s32(v_c0, vsubl_s16(v_Cr, v_delta));
            v_r1 = vaddw_s16(vshrq_n_s32(vaddq_s32(v_r1, v_delta2), yuv_shift), v_Y);

            uint8x8_t v_b = vqmovun_s16(vcombine_s16(vmovn_s32(v_b0), vmovn_s32(v_b1)));
            uint8x8_t v_g = vqmovun_s16(vcombine_s16(vmovn_s32(v_g0), vmovn_s32(v_g1)));
            uint8x8_t v_r = vqmovun_s16(vcombine_s16(vmovn_s32(v_r0), vmovn_s32(v_r1)));

            if (dcn == 3)
            {
                uint8x8x3_t v_dst;
                v_dst.val[bidx] = v_b;
                v_dst.val[1] = v_g;
                v_dst.val[bidx^2] = v_r;
                vst3_u8(dst, v_dst);
            }
            else
            {
                uint8x8x4_t v_dst;
                v_dst.val[bidx] = v_b;
                v_dst.val[1] = v_g;
                v_dst.val[bidx^2] = v_r;
                v_dst.val[3] = v_alpha;
                vst4_u8(dst, v_dst);
            }
        }

        for ( ; i < n; i += 3, dst += dcn)
        {
            uchar Y = src[i];
            uchar Cr = src[i+1+yuvOrder];
            uchar Cb = src[i+2-yuvOrder];

            int b = Y + CV_DESCALE((Cb - delta)*C3, yuv_shift);
            int g = Y + CV_DESCALE((Cb - delta)*C2 + (Cr - delta)*C1, yuv_shift);
            int r = Y + CV_DESCALE((Cr - delta)*C0, yuv_shift);

            dst[bidx] = saturate_cast<uchar>(b);
            dst[1] = saturate_cast<uchar>(g);
            dst[bidx^2] = saturate_cast<uchar>(r);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    bool isCrCb;
    int coeffs[4];

    int32x4_t v_c0, v_c1, v_c2, v_c3, v_delta2;
    int16x4_t v_delta;
    uint8x8_t v_alpha;
};

template <>
struct YCrCb2RGB_i<ushort>
{
    typedef ushort channel_type;

    YCrCb2RGB_i(int _dstcn, int _blueIdx, bool _isCrCb)
        : dstcn(_dstcn), blueIdx(_blueIdx), isCrCb(_isCrCb)
    {
        static const int coeffs_crb[] = { CR2RI, CR2GI, CB2GI, CB2BI};
        static const int coeffs_yuv[] = {  V2RI,  V2GI,  U2GI, U2BI };
        memcpy(coeffs, isCrCb ? coeffs_crb : coeffs_yuv, 4*sizeof(coeffs[0]));

        v_c0 = vdupq_n_s32(coeffs[0]);
        v_c1 = vdupq_n_s32(coeffs[1]);
        v_c2 = vdupq_n_s32(coeffs[2]);
        v_c3 = vdupq_n_s32(coeffs[3]);
        v_delta = vdupq_n_s32(ColorChannel<ushort>::half());
        v_delta2 = vdupq_n_s32(1 << (yuv_shift - 1));
        v_alpha = vdupq_n_u16(ColorChannel<ushort>::max());
        v_alpha2 = vget_low_u16(v_alpha);
    }

    void operator()(const ushort* src, ushort* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx, i = 0;
        int yuvOrder = !isCrCb; //1 if YUV, 0 if YCrCb
        const ushort delta = ColorChannel<ushort>::half(), alpha = ColorChannel<ushort>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3];
        n *= 3;

        for ( ; i <= n - 24; i += 24, dst += dcn * 8)
        {
            uint16x8x3_t v_src = vld3q_u16(src + i);

            int32x4_t v_Y = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src.val[0]))),
                      v_Cr = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src.val[1+yuvOrder]))),
                      v_Cb = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src.val[2-yuvOrder])));

            int32x4_t v_b0 = vmulq_s32(v_c3, vsubq_s32(v_Cb, v_delta));
            v_b0 = vaddq_s32(vshrq_n_s32(vaddq_s32(v_b0, v_delta2), yuv_shift), v_Y);
            int32x4_t v_g0 = vmlaq_s32(vmulq_s32(vsubq_s32(v_Cr, v_delta), v_c1), vsubq_s32(v_Cb, v_delta), v_c2);
            v_g0 = vaddq_s32(vshrq_n_s32(vaddq_s32(v_g0, v_delta2), yuv_shift), v_Y);
            int32x4_t v_r0 = vmulq_s32(v_c0, vsubq_s32(v_Cr, v_delta));
            v_r0 = vaddq_s32(vshrq_n_s32(vaddq_s32(v_r0, v_delta2), yuv_shift), v_Y);

            v_Y = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src.val[0]))),
            v_Cr = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src.val[1+yuvOrder]))),
            v_Cb = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src.val[2-yuvOrder])));

            int32x4_t v_b1 = vmulq_s32(v_c3, vsubq_s32(v_Cb, v_delta));
            v_b1 = vaddq_s32(vshrq_n_s32(vaddq_s32(v_b1, v_delta2), yuv_shift), v_Y);
            int32x4_t v_g1 = vmlaq_s32(vmulq_s32(vsubq_s32(v_Cr, v_delta), v_c1), vsubq_s32(v_Cb, v_delta), v_c2);
            v_g1 = vaddq_s32(vshrq_n_s32(vaddq_s32(v_g1, v_delta2), yuv_shift), v_Y);
            int32x4_t v_r1 = vmulq_s32(v_c0, vsubq_s32(v_Cr, v_delta));
            v_r1 = vaddq_s32(vshrq_n_s32(vaddq_s32(v_r1, v_delta2), yuv_shift), v_Y);

            uint16x8_t v_b = vcombine_u16(vqmovun_s32(v_b0), vqmovun_s32(v_b1));
            uint16x8_t v_g = vcombine_u16(vqmovun_s32(v_g0), vqmovun_s32(v_g1));
            uint16x8_t v_r = vcombine_u16(vqmovun_s32(v_r0), vqmovun_s32(v_r1));

            if (dcn == 3)
            {
                uint16x8x3_t v_dst;
                v_dst.val[bidx] = v_b;
                v_dst.val[1] = v_g;
                v_dst.val[bidx^2] = v_r;
                vst3q_u16(dst, v_dst);
            }
            else
            {
                uint16x8x4_t v_dst;
                v_dst.val[bidx] = v_b;
                v_dst.val[1] = v_g;
                v_dst.val[bidx^2] = v_r;
                v_dst.val[3] = v_alpha;
                vst4q_u16(dst, v_dst);
            }
        }

        for ( ; i <= n - 12; i += 12, dst += dcn * 4)
        {
            uint16x4x3_t v_src = vld3_u16(src + i);

            int32x4_t v_Y = vreinterpretq_s32_u32(vmovl_u16(v_src.val[0])),
                      v_Cr = vreinterpretq_s32_u32(vmovl_u16(v_src.val[1+yuvOrder])),
                      v_Cb = vreinterpretq_s32_u32(vmovl_u16(v_src.val[2-yuvOrder]));

            int32x4_t v_b = vmulq_s32(v_c3, vsubq_s32(v_Cb, v_delta));
            v_b = vaddq_s32(vshrq_n_s32(vaddq_s32(v_b, v_delta2), yuv_shift), v_Y);
            int32x4_t v_g = vmlaq_s32(vmulq_s32(vsubq_s32(v_Cr, v_delta), v_c1), vsubq_s32(v_Cb, v_delta), v_c2);
            v_g = vaddq_s32(vshrq_n_s32(vaddq_s32(v_g, v_delta2), yuv_shift), v_Y);
            int32x4_t v_r = vmulq_s32(vsubq_s32(v_Cr, v_delta), v_c0);
            v_r = vaddq_s32(vshrq_n_s32(vaddq_s32(v_r, v_delta2), yuv_shift), v_Y);

            uint16x4_t v_bd = vqmovun_s32(v_b);
            uint16x4_t v_gd = vqmovun_s32(v_g);
            uint16x4_t v_rd = vqmovun_s32(v_r);

            if (dcn == 3)
            {
                uint16x4x3_t v_dst;
                v_dst.val[bidx] = v_bd;
                v_dst.val[1] = v_gd;
                v_dst.val[bidx^2] = v_rd;
                vst3_u16(dst, v_dst);
            }
            else
            {
                uint16x4x4_t v_dst;
                v_dst.val[bidx] = v_bd;
                v_dst.val[1] = v_gd;
                v_dst.val[bidx^2] = v_rd;
                v_dst.val[3] = v_alpha2;
                vst4_u16(dst, v_dst);
            }
        }

        for ( ; i < n; i += 3, dst += dcn)
        {
            ushort Y = src[i];
            ushort Cr = src[i+1+yuvOrder];
            ushort Cb = src[i+2-yuvOrder];

            int b = Y + CV_DESCALE((Cb - delta)*C3, yuv_shift);
            int g = Y + CV_DESCALE((Cb - delta)*C2 + (Cr - delta)*C1, yuv_shift);
            int r = Y + CV_DESCALE((Cr - delta)*C0, yuv_shift);

            dst[bidx] = saturate_cast<ushort>(b);
            dst[1] = saturate_cast<ushort>(g);
            dst[bidx^2] = saturate_cast<ushort>(r);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    bool isCrCb;
    int coeffs[4];

    int32x4_t v_c0, v_c1, v_c2, v_c3, v_delta2, v_delta;
    uint16x8_t v_alpha;
    uint16x4_t v_alpha2;
};

#endif // CV_SSE2

////////////////////////////////////// RGB <-> XYZ ///////////////////////////////////////

static const float sRGB2XYZ_D65[] =
{
    0.412453f, 0.357580f, 0.180423f,
    0.212671f, 0.715160f, 0.072169f,
    0.019334f, 0.119193f, 0.950227f
};

static const float XYZ2sRGB_D65[] =
{
    3.240479f, -1.53715f, -0.498535f,
    -0.969256f, 1.875991f, 0.041556f,
    0.055648f, -0.204043f, 1.057311f
};

template<typename _Tp> struct RGB2XYZ_f
{
    typedef _Tp channel_type;

    RGB2XYZ_f(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        memcpy(coeffs, _coeffs ? _coeffs : sRGB2XYZ_D65, 9*sizeof(coeffs[0]));
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

        n *= 3;
        for(int i = 0; i < n; i += 3, src += scn)
        {
            _Tp X = saturate_cast<_Tp>(src[0]*C0 + src[1]*C1 + src[2]*C2);
            _Tp Y = saturate_cast<_Tp>(src[0]*C3 + src[1]*C4 + src[2]*C5);
            _Tp Z = saturate_cast<_Tp>(src[0]*C6 + src[1]*C7 + src[2]*C8);
            dst[i] = X; dst[i+1] = Y; dst[i+2] = Z;
        }
    }
    int srccn;
    float coeffs[9];
};

#if CV_NEON

template <>
struct RGB2XYZ_f<float>
{
    typedef float channel_type;

    RGB2XYZ_f(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        memcpy(coeffs, _coeffs ? _coeffs : sRGB2XYZ_D65, 9*sizeof(coeffs[0]));
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }

        v_c0 = vdupq_n_f32(coeffs[0]);
        v_c1 = vdupq_n_f32(coeffs[1]);
        v_c2 = vdupq_n_f32(coeffs[2]);
        v_c3 = vdupq_n_f32(coeffs[3]);
        v_c4 = vdupq_n_f32(coeffs[4]);
        v_c5 = vdupq_n_f32(coeffs[5]);
        v_c6 = vdupq_n_f32(coeffs[6]);
        v_c7 = vdupq_n_f32(coeffs[7]);
        v_c8 = vdupq_n_f32(coeffs[8]);
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int scn = srccn, i = 0;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

        n *= 3;

        if (scn == 3)
            for ( ; i <= n - 12; i += 12, src += 12)
            {
                float32x4x3_t v_src = vld3q_f32(src), v_dst;
                v_dst.val[0] = vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_c0), v_src.val[1], v_c1), v_src.val[2], v_c2);
                v_dst.val[1] = vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_c3), v_src.val[1], v_c4), v_src.val[2], v_c5);
                v_dst.val[2] = vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_c6), v_src.val[1], v_c7), v_src.val[2], v_c8);
                vst3q_f32(dst + i, v_dst);
            }
        else
            for ( ; i <= n - 12; i += 12, src += 16)
            {
                float32x4x4_t v_src = vld4q_f32(src);
                float32x4x3_t v_dst;
                v_dst.val[0] = vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_c0), v_src.val[1], v_c1), v_src.val[2], v_c2);
                v_dst.val[1] = vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_c3), v_src.val[1], v_c4), v_src.val[2], v_c5);
                v_dst.val[2] = vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_c6), v_src.val[1], v_c7), v_src.val[2], v_c8);
                vst3q_f32(dst + i, v_dst);
            }

        for ( ; i < n; i += 3, src += scn)
        {
            float X = saturate_cast<float>(src[0]*C0 + src[1]*C1 + src[2]*C2);
            float Y = saturate_cast<float>(src[0]*C3 + src[1]*C4 + src[2]*C5);
            float Z = saturate_cast<float>(src[0]*C6 + src[1]*C7 + src[2]*C8);
            dst[i] = X; dst[i+1] = Y; dst[i+2] = Z;
        }
    }

    int srccn;
    float coeffs[9];
    float32x4_t v_c0, v_c1, v_c2, v_c3, v_c4, v_c5, v_c6, v_c7, v_c8;
};

#endif

template<typename _Tp> struct RGB2XYZ_i
{
    typedef _Tp channel_type;

    RGB2XYZ_i(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        static const int coeffs0[] =
        {
            1689,    1465,    739,
            871,     2929,    296,
            79,      488,     3892
        };
        for( int i = 0; i < 9; i++ )
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << xyz_shift)) : coeffs0[i];
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn;
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;

        for(int i = 0; i < n; i += 3, src += scn)
        {
            int X = CV_DESCALE(src[0]*C0 + src[1]*C1 + src[2]*C2, xyz_shift);
            int Y = CV_DESCALE(src[0]*C3 + src[1]*C4 + src[2]*C5, xyz_shift);
            int Z = CV_DESCALE(src[0]*C6 + src[1]*C7 + src[2]*C8, xyz_shift);
            dst[i] = saturate_cast<_Tp>(X); dst[i+1] = saturate_cast<_Tp>(Y);
            dst[i+2] = saturate_cast<_Tp>(Z);
        }
    }
    int srccn;
    int coeffs[9];
};

#if CV_NEON

template <>
struct RGB2XYZ_i<uchar>
{
    typedef uchar channel_type;

    RGB2XYZ_i(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        static const int coeffs0[] =
        {
            1689,    1465,    739,
            871,     2929,    296,
            79,      488,     3892
        };
        for( int i = 0; i < 9; i++ )
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << xyz_shift)) : coeffs0[i];
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }

        v_c0 = vdup_n_u16(coeffs[0]);
        v_c1 = vdup_n_u16(coeffs[1]);
        v_c2 = vdup_n_u16(coeffs[2]);
        v_c3 = vdup_n_u16(coeffs[3]);
        v_c4 = vdup_n_u16(coeffs[4]);
        v_c5 = vdup_n_u16(coeffs[5]);
        v_c6 = vdup_n_u16(coeffs[6]);
        v_c7 = vdup_n_u16(coeffs[7]);
        v_c8 = vdup_n_u16(coeffs[8]);
        v_delta = vdupq_n_u32(1 << (xyz_shift - 1));
    }
    void operator()(const uchar * src, uchar * dst, int n) const
    {
        int scn = srccn, i = 0;
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;

        for ( ; i <= n - 24; i += 24, src += scn * 8)
        {
            uint8x8x3_t v_dst;
            uint16x8x3_t v_src16;

            if (scn == 3)
            {
                uint8x8x3_t v_src = vld3_u8(src);
                v_src16.val[0] = vmovl_u8(v_src.val[0]);
                v_src16.val[1] = vmovl_u8(v_src.val[1]);
                v_src16.val[2] = vmovl_u8(v_src.val[2]);
            }
            else
            {
                uint8x8x4_t v_src = vld4_u8(src);
                v_src16.val[0] = vmovl_u8(v_src.val[0]);
                v_src16.val[1] = vmovl_u8(v_src.val[1]);
                v_src16.val[2] = vmovl_u8(v_src.val[2]);
            }

            uint16x4_t v_s0 = vget_low_u16(v_src16.val[0]),
                       v_s1 = vget_low_u16(v_src16.val[1]),
                       v_s2 = vget_low_u16(v_src16.val[2]);

            uint32x4_t v_X0 = vmlal_u16(vmlal_u16(vmull_u16(v_s0, v_c0), v_s1, v_c1), v_s2, v_c2);
            uint32x4_t v_Y0 = vmlal_u16(vmlal_u16(vmull_u16(v_s0, v_c3), v_s1, v_c4), v_s2, v_c5);
            uint32x4_t v_Z0 = vmlal_u16(vmlal_u16(vmull_u16(v_s0, v_c6), v_s1, v_c7), v_s2, v_c8);
            v_X0 = vshrq_n_u32(vaddq_u32(v_X0, v_delta), xyz_shift);
            v_Y0 = vshrq_n_u32(vaddq_u32(v_Y0, v_delta), xyz_shift);
            v_Z0 = vshrq_n_u32(vaddq_u32(v_Z0, v_delta), xyz_shift);

            v_s0 = vget_high_u16(v_src16.val[0]),
            v_s1 = vget_high_u16(v_src16.val[1]),
            v_s2 = vget_high_u16(v_src16.val[2]);

            uint32x4_t v_X1 = vmlal_u16(vmlal_u16(vmull_u16(v_s0, v_c0), v_s1, v_c1), v_s2, v_c2);
            uint32x4_t v_Y1 = vmlal_u16(vmlal_u16(vmull_u16(v_s0, v_c3), v_s1, v_c4), v_s2, v_c5);
            uint32x4_t v_Z1 = vmlal_u16(vmlal_u16(vmull_u16(v_s0, v_c6), v_s1, v_c7), v_s2, v_c8);
            v_X1 = vshrq_n_u32(vaddq_u32(v_X1, v_delta), xyz_shift);
            v_Y1 = vshrq_n_u32(vaddq_u32(v_Y1, v_delta), xyz_shift);
            v_Z1 = vshrq_n_u32(vaddq_u32(v_Z1, v_delta), xyz_shift);

            v_dst.val[0] = vqmovn_u16(vcombine_u16(vmovn_u32(v_X0), vmovn_u32(v_X1)));
            v_dst.val[1] = vqmovn_u16(vcombine_u16(vmovn_u32(v_Y0), vmovn_u32(v_Y1)));
            v_dst.val[2] = vqmovn_u16(vcombine_u16(vmovn_u32(v_Z0), vmovn_u32(v_Z1)));

            vst3_u8(dst + i, v_dst);
        }

        for ( ; i < n; i += 3, src += scn)
        {
            int X = CV_DESCALE(src[0]*C0 + src[1]*C1 + src[2]*C2, xyz_shift);
            int Y = CV_DESCALE(src[0]*C3 + src[1]*C4 + src[2]*C5, xyz_shift);
            int Z = CV_DESCALE(src[0]*C6 + src[1]*C7 + src[2]*C8, xyz_shift);
            dst[i] = saturate_cast<uchar>(X);
            dst[i+1] = saturate_cast<uchar>(Y);
            dst[i+2] = saturate_cast<uchar>(Z);
        }
    }

    int srccn, coeffs[9];
    uint16x4_t v_c0, v_c1, v_c2, v_c3, v_c4, v_c5, v_c6, v_c7, v_c8;
    uint32x4_t v_delta;
};

template <>
struct RGB2XYZ_i<ushort>
{
    typedef ushort channel_type;

    RGB2XYZ_i(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        static const int coeffs0[] =
        {
            1689,    1465,    739,
            871,     2929,    296,
            79,      488,     3892
        };
        for( int i = 0; i < 9; i++ )
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << xyz_shift)) : coeffs0[i];
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }

        v_c0 = vdup_n_u16(coeffs[0]);
        v_c1 = vdup_n_u16(coeffs[1]);
        v_c2 = vdup_n_u16(coeffs[2]);
        v_c3 = vdup_n_u16(coeffs[3]);
        v_c4 = vdup_n_u16(coeffs[4]);
        v_c5 = vdup_n_u16(coeffs[5]);
        v_c6 = vdup_n_u16(coeffs[6]);
        v_c7 = vdup_n_u16(coeffs[7]);
        v_c8 = vdup_n_u16(coeffs[8]);
        v_delta = vdupq_n_u32(1 << (xyz_shift - 1));
    }

    void operator()(const ushort * src, ushort * dst, int n) const
    {
        int scn = srccn, i = 0;
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;

        for ( ; i <= n - 24; i += 24, src += scn * 8)
        {
            uint16x8x3_t v_src, v_dst;

            if (scn == 3)
                v_src = vld3q_u16(src);
            else
            {
                uint16x8x4_t v_src4 = vld4q_u16(src);
                v_src.val[0] = v_src4.val[0];
                v_src.val[1] = v_src4.val[1];
                v_src.val[2] = v_src4.val[2];
            }

            uint16x4_t v_s0 = vget_low_u16(v_src.val[0]),
                       v_s1 = vget_low_u16(v_src.val[1]),
                       v_s2 = vget_low_u16(v_src.val[2]);

            uint32x4_t v_X0 = vmlal_u16(vmlal_u16(vmull_u16(v_s0, v_c0), v_s1, v_c1), v_s2, v_c2);
            uint32x4_t v_Y0 = vmlal_u16(vmlal_u16(vmull_u16(v_s0, v_c3), v_s1, v_c4), v_s2, v_c5);
            uint32x4_t v_Z0 = vmlal_u16(vmlal_u16(vmull_u16(v_s0, v_c6), v_s1, v_c7), v_s2, v_c8);
            v_X0 = vshrq_n_u32(vaddq_u32(v_X0, v_delta), xyz_shift);
            v_Y0 = vshrq_n_u32(vaddq_u32(v_Y0, v_delta), xyz_shift);
            v_Z0 = vshrq_n_u32(vaddq_u32(v_Z0, v_delta), xyz_shift);

            v_s0 = vget_high_u16(v_src.val[0]),
            v_s1 = vget_high_u16(v_src.val[1]),
            v_s2 = vget_high_u16(v_src.val[2]);

            uint32x4_t v_X1 = vmlal_u16(vmlal_u16(vmull_u16(v_s0, v_c0), v_s1, v_c1), v_s2, v_c2);
            uint32x4_t v_Y1 = vmlal_u16(vmlal_u16(vmull_u16(v_s0, v_c3), v_s1, v_c4), v_s2, v_c5);
            uint32x4_t v_Z1 = vmlal_u16(vmlal_u16(vmull_u16(v_s0, v_c6), v_s1, v_c7), v_s2, v_c8);
            v_X1 = vshrq_n_u32(vaddq_u32(v_X1, v_delta), xyz_shift);
            v_Y1 = vshrq_n_u32(vaddq_u32(v_Y1, v_delta), xyz_shift);
            v_Z1 = vshrq_n_u32(vaddq_u32(v_Z1, v_delta), xyz_shift);

            v_dst.val[0] = vcombine_u16(vqmovn_u32(v_X0), vqmovn_u32(v_X1));
            v_dst.val[1] = vcombine_u16(vqmovn_u32(v_Y0), vqmovn_u32(v_Y1));
            v_dst.val[2] = vcombine_u16(vqmovn_u32(v_Z0), vqmovn_u32(v_Z1));

            vst3q_u16(dst + i, v_dst);
        }

        for ( ; i <= n - 12; i += 12, src += scn * 4)
        {
            uint16x4x3_t v_dst;
            uint16x4_t v_s0, v_s1, v_s2;

            if (scn == 3)
            {
                uint16x4x3_t v_src = vld3_u16(src);
                v_s0 = v_src.val[0];
                v_s1 = v_src.val[1];
                v_s2 = v_src.val[2];
            }
            else
            {
                uint16x4x4_t v_src = vld4_u16(src);
                v_s0 = v_src.val[0];
                v_s1 = v_src.val[1];
                v_s2 = v_src.val[2];
            }

            uint32x4_t v_X = vmlal_u16(vmlal_u16(vmull_u16(v_s0, v_c0), v_s1, v_c1), v_s2, v_c2);
            uint32x4_t v_Y = vmlal_u16(vmlal_u16(vmull_u16(v_s0, v_c3), v_s1, v_c4), v_s2, v_c5);
            uint32x4_t v_Z = vmlal_u16(vmlal_u16(vmull_u16(v_s0, v_c6), v_s1, v_c7), v_s2, v_c8);

            v_dst.val[0] = vqmovn_u32(vshrq_n_u32(vaddq_u32(v_X, v_delta), xyz_shift));
            v_dst.val[1] = vqmovn_u32(vshrq_n_u32(vaddq_u32(v_Y, v_delta), xyz_shift));
            v_dst.val[2] = vqmovn_u32(vshrq_n_u32(vaddq_u32(v_Z, v_delta), xyz_shift));

            vst3_u16(dst + i, v_dst);
        }

        for ( ; i < n; i += 3, src += scn)
        {
            int X = CV_DESCALE(src[0]*C0 + src[1]*C1 + src[2]*C2, xyz_shift);
            int Y = CV_DESCALE(src[0]*C3 + src[1]*C4 + src[2]*C5, xyz_shift);
            int Z = CV_DESCALE(src[0]*C6 + src[1]*C7 + src[2]*C8, xyz_shift);
            dst[i] = saturate_cast<ushort>(X);
            dst[i+1] = saturate_cast<ushort>(Y);
            dst[i+2] = saturate_cast<ushort>(Z);
        }
    }

    int srccn, coeffs[9];
    uint16x4_t v_c0, v_c1, v_c2, v_c3, v_c4, v_c5, v_c6, v_c7, v_c8;
    uint32x4_t v_delta;
};

#endif

template<typename _Tp> struct XYZ2RGB_f
{
    typedef _Tp channel_type;

    XYZ2RGB_f(int _dstcn, int _blueIdx, const float* _coeffs)
    : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        memcpy(coeffs, _coeffs ? _coeffs : XYZ2sRGB_D65, 9*sizeof(coeffs[0]));
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }
    }

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int dcn = dstcn;
        _Tp alpha = ColorChannel<_Tp>::max();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;
        for(int i = 0; i < n; i += 3, dst += dcn)
        {
            _Tp B = saturate_cast<_Tp>(src[i]*C0 + src[i+1]*C1 + src[i+2]*C2);
            _Tp G = saturate_cast<_Tp>(src[i]*C3 + src[i+1]*C4 + src[i+2]*C5);
            _Tp R = saturate_cast<_Tp>(src[i]*C6 + src[i+1]*C7 + src[i+2]*C8);
            dst[0] = B; dst[1] = G; dst[2] = R;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    float coeffs[9];
};

template<typename _Tp> struct XYZ2RGB_i
{
    typedef _Tp channel_type;

    XYZ2RGB_i(int _dstcn, int _blueIdx, const int* _coeffs)
    : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        static const int coeffs0[] =
        {
            13273,  -6296,  -2042,
            -3970,   7684,    170,
              228,   -836,   4331
        };
        for(int i = 0; i < 9; i++)
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << xyz_shift)) : coeffs0[i];

        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int dcn = dstcn;
        _Tp alpha = ColorChannel<_Tp>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;
        for(int i = 0; i < n; i += 3, dst += dcn)
        {
            int B = CV_DESCALE(src[i]*C0 + src[i+1]*C1 + src[i+2]*C2, xyz_shift);
            int G = CV_DESCALE(src[i]*C3 + src[i+1]*C4 + src[i+2]*C5, xyz_shift);
            int R = CV_DESCALE(src[i]*C6 + src[i+1]*C7 + src[i+2]*C8, xyz_shift);
            dst[0] = saturate_cast<_Tp>(B); dst[1] = saturate_cast<_Tp>(G);
            dst[2] = saturate_cast<_Tp>(R);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    int coeffs[9];
};

#if CV_NEON

template <>
struct XYZ2RGB_i<uchar>
{
    typedef uchar channel_type;

    XYZ2RGB_i(int _dstcn, int _blueIdx, const int* _coeffs)
    : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        static const int coeffs0[] =
        {
            13273,  -6296,  -2042,
            -3970,   7684,    170,
              228,   -836,   4331
        };
        for(int i = 0; i < 9; i++)
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << xyz_shift)) : coeffs0[i];

        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }

        v_c0 = vdup_n_s16(coeffs[0]);
        v_c1 = vdup_n_s16(coeffs[1]);
        v_c2 = vdup_n_s16(coeffs[2]);
        v_c3 = vdup_n_s16(coeffs[3]);
        v_c4 = vdup_n_s16(coeffs[4]);
        v_c5 = vdup_n_s16(coeffs[5]);
        v_c6 = vdup_n_s16(coeffs[6]);
        v_c7 = vdup_n_s16(coeffs[7]);
        v_c8 = vdup_n_s16(coeffs[8]);
        v_delta = vdupq_n_s32(1 << (xyz_shift - 1));
        v_alpha = vmovn_u16(vdupq_n_u16(ColorChannel<uchar>::max()));
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int dcn = dstcn, i = 0;
        uchar alpha = ColorChannel<uchar>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;

        for ( ; i <= n - 24; i += 24, dst += dcn * 8)
        {
            uint8x8x3_t v_src = vld3_u8(src + i);
            int16x8x3_t v_src16;
            v_src16.val[0] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[0]));
            v_src16.val[1] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[1]));
            v_src16.val[2] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[2]));

            int16x4_t v_s0 = vget_low_s16(v_src16.val[0]),
                       v_s1 = vget_low_s16(v_src16.val[1]),
                       v_s2 = vget_low_s16(v_src16.val[2]);

            int32x4_t v_X0 = vmlal_s16(vmlal_s16(vmull_s16(v_s0, v_c0), v_s1, v_c1), v_s2, v_c2);
            int32x4_t v_Y0 = vmlal_s16(vmlal_s16(vmull_s16(v_s0, v_c3), v_s1, v_c4), v_s2, v_c5);
            int32x4_t v_Z0 = vmlal_s16(vmlal_s16(vmull_s16(v_s0, v_c6), v_s1, v_c7), v_s2, v_c8);
            v_X0 = vshrq_n_s32(vaddq_s32(v_X0, v_delta), xyz_shift);
            v_Y0 = vshrq_n_s32(vaddq_s32(v_Y0, v_delta), xyz_shift);
            v_Z0 = vshrq_n_s32(vaddq_s32(v_Z0, v_delta), xyz_shift);

            v_s0 = vget_high_s16(v_src16.val[0]),
            v_s1 = vget_high_s16(v_src16.val[1]),
            v_s2 = vget_high_s16(v_src16.val[2]);

            int32x4_t v_X1 = vmlal_s16(vmlal_s16(vmull_s16(v_s0, v_c0), v_s1, v_c1), v_s2, v_c2);
            int32x4_t v_Y1 = vmlal_s16(vmlal_s16(vmull_s16(v_s0, v_c3), v_s1, v_c4), v_s2, v_c5);
            int32x4_t v_Z1 = vmlal_s16(vmlal_s16(vmull_s16(v_s0, v_c6), v_s1, v_c7), v_s2, v_c8);
            v_X1 = vshrq_n_s32(vaddq_s32(v_X1, v_delta), xyz_shift);
            v_Y1 = vshrq_n_s32(vaddq_s32(v_Y1, v_delta), xyz_shift);
            v_Z1 = vshrq_n_s32(vaddq_s32(v_Z1, v_delta), xyz_shift);

            uint8x8_t v_b = vqmovun_s16(vcombine_s16(vqmovn_s32(v_X0), vqmovn_s32(v_X1)));
            uint8x8_t v_g = vqmovun_s16(vcombine_s16(vqmovn_s32(v_Y0), vqmovn_s32(v_Y1)));
            uint8x8_t v_r = vqmovun_s16(vcombine_s16(vqmovn_s32(v_Z0), vqmovn_s32(v_Z1)));

            if (dcn == 3)
            {
                uint8x8x3_t v_dst;
                v_dst.val[0] = v_b;
                v_dst.val[1] = v_g;
                v_dst.val[2] = v_r;
                vst3_u8(dst, v_dst);
            }
            else
            {
                uint8x8x4_t v_dst;
                v_dst.val[0] = v_b;
                v_dst.val[1] = v_g;
                v_dst.val[2] = v_r;
                v_dst.val[3] = v_alpha;
                vst4_u8(dst, v_dst);
            }
        }

        for ( ; i < n; i += 3, dst += dcn)
        {
            int B = CV_DESCALE(src[i]*C0 + src[i+1]*C1 + src[i+2]*C2, xyz_shift);
            int G = CV_DESCALE(src[i]*C3 + src[i+1]*C4 + src[i+2]*C5, xyz_shift);
            int R = CV_DESCALE(src[i]*C6 + src[i+1]*C7 + src[i+2]*C8, xyz_shift);
            dst[0] = saturate_cast<uchar>(B); dst[1] = saturate_cast<uchar>(G);
            dst[2] = saturate_cast<uchar>(R);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    int coeffs[9];

    int16x4_t v_c0, v_c1, v_c2, v_c3, v_c4, v_c5, v_c6, v_c7, v_c8;
    uint8x8_t v_alpha;
    int32x4_t v_delta;
};

template <>
struct XYZ2RGB_i<ushort>
{
    typedef ushort channel_type;

    XYZ2RGB_i(int _dstcn, int _blueIdx, const int* _coeffs)
    : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        static const int coeffs0[] =
        {
            13273,  -6296,  -2042,
            -3970,   7684,    170,
              228,   -836,   4331
        };
        for(int i = 0; i < 9; i++)
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << xyz_shift)) : coeffs0[i];

        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }

        v_c0 = vdupq_n_s32(coeffs[0]);
        v_c1 = vdupq_n_s32(coeffs[1]);
        v_c2 = vdupq_n_s32(coeffs[2]);
        v_c3 = vdupq_n_s32(coeffs[3]);
        v_c4 = vdupq_n_s32(coeffs[4]);
        v_c5 = vdupq_n_s32(coeffs[5]);
        v_c6 = vdupq_n_s32(coeffs[6]);
        v_c7 = vdupq_n_s32(coeffs[7]);
        v_c8 = vdupq_n_s32(coeffs[8]);
        v_delta = vdupq_n_s32(1 << (xyz_shift - 1));
        v_alpha = vdupq_n_u16(ColorChannel<ushort>::max());
        v_alpha2 = vget_low_u16(v_alpha);
    }

    void operator()(const ushort* src, ushort* dst, int n) const
    {
        int dcn = dstcn, i = 0;
        ushort alpha = ColorChannel<ushort>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;

        for ( ; i <= n - 24; i += 24, dst += dcn * 8)
        {
            uint16x8x3_t v_src = vld3q_u16(src + i);
            int32x4_t v_s0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src.val[0]))),
                      v_s1 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src.val[1]))),
                      v_s2 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src.val[2])));

            int32x4_t v_X0 = vmlaq_s32(vmlaq_s32(vmulq_s32(v_s0, v_c0), v_s1, v_c1), v_s2, v_c2);
            int32x4_t v_Y0 = vmlaq_s32(vmlaq_s32(vmulq_s32(v_s0, v_c3), v_s1, v_c4), v_s2, v_c5);
            int32x4_t v_Z0 = vmlaq_s32(vmlaq_s32(vmulq_s32(v_s0, v_c6), v_s1, v_c7), v_s2, v_c8);
            v_X0 = vshrq_n_s32(vaddq_s32(v_X0, v_delta), xyz_shift);
            v_Y0 = vshrq_n_s32(vaddq_s32(v_Y0, v_delta), xyz_shift);
            v_Z0 = vshrq_n_s32(vaddq_s32(v_Z0, v_delta), xyz_shift);

            v_s0 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src.val[0])));
            v_s1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src.val[1])));
            v_s2 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src.val[2])));

            int32x4_t v_X1 = vmlaq_s32(vmlaq_s32(vmulq_s32(v_s0, v_c0), v_s1, v_c1), v_s2, v_c2);
            int32x4_t v_Y1 = vmlaq_s32(vmlaq_s32(vmulq_s32(v_s0, v_c3), v_s1, v_c4), v_s2, v_c5);
            int32x4_t v_Z1 = vmlaq_s32(vmlaq_s32(vmulq_s32(v_s0, v_c6), v_s1, v_c7), v_s2, v_c8);
            v_X1 = vshrq_n_s32(vaddq_s32(v_X1, v_delta), xyz_shift);
            v_Y1 = vshrq_n_s32(vaddq_s32(v_Y1, v_delta), xyz_shift);
            v_Z1 = vshrq_n_s32(vaddq_s32(v_Z1, v_delta), xyz_shift);

            uint16x8_t v_b = vcombine_u16(vqmovun_s32(v_X0), vqmovun_s32(v_X1));
            uint16x8_t v_g = vcombine_u16(vqmovun_s32(v_Y0), vqmovun_s32(v_Y1));
            uint16x8_t v_r = vcombine_u16(vqmovun_s32(v_Z0), vqmovun_s32(v_Z1));

            if (dcn == 3)
            {
                uint16x8x3_t v_dst;
                v_dst.val[0] = v_b;
                v_dst.val[1] = v_g;
                v_dst.val[2] = v_r;
                vst3q_u16(dst, v_dst);
            }
            else
            {
                uint16x8x4_t v_dst;
                v_dst.val[0] = v_b;
                v_dst.val[1] = v_g;
                v_dst.val[2] = v_r;
                v_dst.val[3] = v_alpha;
                vst4q_u16(dst, v_dst);
            }
        }

        for ( ; i <= n - 12; i += 12, dst += dcn * 4)
        {
            uint16x4x3_t v_src = vld3_u16(src + i);
            int32x4_t v_s0 = vreinterpretq_s32_u32(vmovl_u16(v_src.val[0])),
                      v_s1 = vreinterpretq_s32_u32(vmovl_u16(v_src.val[1])),
                      v_s2 = vreinterpretq_s32_u32(vmovl_u16(v_src.val[2]));

            int32x4_t v_X = vmlaq_s32(vmlaq_s32(vmulq_s32(v_s0, v_c0), v_s1, v_c1), v_s2, v_c2);
            int32x4_t v_Y = vmlaq_s32(vmlaq_s32(vmulq_s32(v_s0, v_c3), v_s1, v_c4), v_s2, v_c5);
            int32x4_t v_Z = vmlaq_s32(vmlaq_s32(vmulq_s32(v_s0, v_c6), v_s1, v_c7), v_s2, v_c8);
            v_X = vshrq_n_s32(vaddq_s32(v_X, v_delta), xyz_shift);
            v_Y = vshrq_n_s32(vaddq_s32(v_Y, v_delta), xyz_shift);
            v_Z = vshrq_n_s32(vaddq_s32(v_Z, v_delta), xyz_shift);

            uint16x4_t v_b = vqmovun_s32(v_X);
            uint16x4_t v_g = vqmovun_s32(v_Y);
            uint16x4_t v_r = vqmovun_s32(v_Z);

            if (dcn == 3)
            {
                uint16x4x3_t v_dst;
                v_dst.val[0] = v_b;
                v_dst.val[1] = v_g;
                v_dst.val[2] = v_r;
                vst3_u16(dst, v_dst);
            }
            else
            {
                uint16x4x4_t v_dst;
                v_dst.val[0] = v_b;
                v_dst.val[1] = v_g;
                v_dst.val[2] = v_r;
                v_dst.val[3] = v_alpha2;
                vst4_u16(dst, v_dst);
            }
        }

        for ( ; i < n; i += 3, dst += dcn)
        {
            int B = CV_DESCALE(src[i]*C0 + src[i+1]*C1 + src[i+2]*C2, xyz_shift);
            int G = CV_DESCALE(src[i]*C3 + src[i+1]*C4 + src[i+2]*C5, xyz_shift);
            int R = CV_DESCALE(src[i]*C6 + src[i+1]*C7 + src[i+2]*C8, xyz_shift);
            dst[0] = saturate_cast<ushort>(B); dst[1] = saturate_cast<ushort>(G);
            dst[2] = saturate_cast<ushort>(R);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    int coeffs[9];

    int32x4_t v_c0, v_c1, v_c2, v_c3, v_c4, v_c5, v_c6, v_c7, v_c8, v_delta;
    uint16x4_t v_alpha2;
    uint16x8_t v_alpha;
};

#endif

////////////////////////////////////// RGB <-> HSV ///////////////////////////////////////


struct RGB2HSV_b
{
    typedef uchar channel_type;

    RGB2HSV_b(int _srccn, int _blueIdx, int _hrange)
    : srccn(_srccn), blueIdx(_blueIdx), hrange(_hrange)
    {
        CV_Assert( hrange == 180 || hrange == 256 );
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, bidx = blueIdx, scn = srccn;
        const int hsv_shift = 12;

        static int sdiv_table[256];
        static int hdiv_table180[256];
        static int hdiv_table256[256];
        static volatile bool initialized = false;

        int hr = hrange;
        const int* hdiv_table = hr == 180 ? hdiv_table180 : hdiv_table256;
        n *= 3;

        if( !initialized )
        {
            sdiv_table[0] = hdiv_table180[0] = hdiv_table256[0] = 0;
            for( i = 1; i < 256; i++ )
            {
                sdiv_table[i] = saturate_cast<int>((255 << hsv_shift)/(1.*i));
                hdiv_table180[i] = saturate_cast<int>((180 << hsv_shift)/(6.*i));
                hdiv_table256[i] = saturate_cast<int>((256 << hsv_shift)/(6.*i));
            }
            initialized = true;
        }

        for( i = 0; i < n; i += 3, src += scn )
        {
            int b = src[bidx], g = src[1], r = src[bidx^2];
            int h, s, v = b;
            int vmin = b, diff;
            int vr, vg;

            CV_CALC_MAX_8U( v, g );
            CV_CALC_MAX_8U( v, r );
            CV_CALC_MIN_8U( vmin, g );
            CV_CALC_MIN_8U( vmin, r );

            diff = v - vmin;
            vr = v == r ? -1 : 0;
            vg = v == g ? -1 : 0;

            s = (diff * sdiv_table[v] + (1 << (hsv_shift-1))) >> hsv_shift;
            h = (vr & (g - b)) +
                (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
            h = (h * hdiv_table[diff] + (1 << (hsv_shift-1))) >> hsv_shift;
            h += h < 0 ? hr : 0;

            dst[i] = saturate_cast<uchar>(h);
            dst[i+1] = (uchar)s;
            dst[i+2] = (uchar)v;
        }
    }

    int srccn, blueIdx, hrange;
};


struct RGB2HSV_f
{
    typedef float channel_type;

    RGB2HSV_f(int _srccn, int _blueIdx, float _hrange)
    : srccn(_srccn), blueIdx(_blueIdx), hrange(_hrange) {}

    void operator()(const float* src, float* dst, int n) const
    {
        int i, bidx = blueIdx, scn = srccn;
        float hscale = hrange*(1.f/360.f);
        n *= 3;

        for( i = 0; i < n; i += 3, src += scn )
        {
            float b = src[bidx], g = src[1], r = src[bidx^2];
            float h, s, v;

            float vmin, diff;

            v = vmin = r;
            if( v < g ) v = g;
            if( v < b ) v = b;
            if( vmin > g ) vmin = g;
            if( vmin > b ) vmin = b;

            diff = v - vmin;
            s = diff/(float)(fabs(v) + FLT_EPSILON);
            diff = (float)(60./(diff + FLT_EPSILON));
            if( v == r )
                h = (g - b)*diff;
            else if( v == g )
                h = (b - r)*diff + 120.f;
            else
                h = (r - g)*diff + 240.f;

            if( h < 0 ) h += 360.f;

            dst[i] = h*hscale;
            dst[i+1] = s;
            dst[i+2] = v;
        }
    }

    int srccn, blueIdx;
    float hrange;
};


struct HSV2RGB_f
{
    typedef float channel_type;

    HSV2RGB_f(int _dstcn, int _blueIdx, float _hrange)
    : dstcn(_dstcn), blueIdx(_blueIdx), hscale(6.f/_hrange) {
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int i = 0, bidx = blueIdx, dcn = dstcn;
        float _hscale = hscale;
        float alpha = ColorChannel<float>::max();
        n *= 3;

        for( ; i < n; i += 3, dst += dcn )
        {
            float h = src[i], s = src[i+1], v = src[i+2];
            float b, g, r;

            if( s == 0 )
                b = g = r = v;
            else
            {
                static const int sector_data[][3]=
                    {{1,3,0}, {1,0,2}, {3,0,1}, {0,2,1}, {0,1,3}, {2,1,0}};
                float tab[4];
                int sector;
                h *= _hscale;
                if( h < 0 )
                    do h += 6; while( h < 0 );
                else if( h >= 6 )
                    do h -= 6; while( h >= 6 );
                sector = cvFloor(h);
                h -= sector;
                if( (unsigned)sector >= 6u )
                {
                    sector = 0;
                    h = 0.f;
                }

                tab[0] = v;
                tab[1] = v*(1.f - s);
                tab[2] = v*(1.f - s*h);
                tab[3] = v*(1.f - s*(1.f - h));

                b = tab[sector_data[sector][0]];
                g = tab[sector_data[sector][1]];
                r = tab[sector_data[sector][2]];
            }

            dst[bidx] = b;
            dst[1] = g;
            dst[bidx^2] = r;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn, blueIdx;
    float hscale;
};


struct HSV2RGB_b
{
    typedef uchar channel_type;

    HSV2RGB_b(int _dstcn, int _blueIdx, int _hrange)
    : dstcn(_dstcn), cvt(3, _blueIdx, (float)_hrange)
    {
        #if CV_NEON
        v_scale_inv = vdupq_n_f32(1.f/255.f);
        v_scale = vdupq_n_f32(255.f);
        v_alpha = vdup_n_u8(ColorChannel<uchar>::max());
        #endif
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();
        float CV_DECL_ALIGNED(16) buf[3*BLOCK_SIZE];

        for( i = 0; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);
            j = 0;

            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24)
            {
                uint8x8x3_t v_src = vld3_u8(src + j);
                uint16x8_t v_t0 = vmovl_u8(v_src.val[0]),
                           v_t1 = vmovl_u8(v_src.val[1]),
                           v_t2 = vmovl_u8(v_src.val[2]);

                float32x4x3_t v_dst;
                v_dst.val[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t0)));
                v_dst.val[1] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t1))), v_scale_inv);
                v_dst.val[2] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t2))), v_scale_inv);
                vst3q_f32(buf + j, v_dst);

                v_dst.val[0] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t0)));
                v_dst.val[1] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t1))), v_scale_inv);
                v_dst.val[2] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t2))), v_scale_inv);
                vst3q_f32(buf + j + 12, v_dst);
            }
            #endif

            for( ; j < dn*3; j += 3 )
            {
                buf[j] = src[j];
                buf[j+1] = src[j+1]*(1.f/255.f);
                buf[j+2] = src[j+2]*(1.f/255.f);
            }
            cvt(buf, buf, dn);

            j = 0;
            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24, dst += dcn * 8)
            {
                float32x4x3_t v_src0 = vld3q_f32(buf + j), v_src1 = vld3q_f32(buf + j + 12);
                uint8x8_t v_dst0 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[0], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[0], v_scale)))));
                uint8x8_t v_dst1 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[1], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[1], v_scale)))));
                uint8x8_t v_dst2 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[2], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[2], v_scale)))));

                if (dcn == 4)
                {
                    uint8x8x4_t v_dst;
                    v_dst.val[0] = v_dst0;
                    v_dst.val[1] = v_dst1;
                    v_dst.val[2] = v_dst2;
                    v_dst.val[3] = v_alpha;
                    vst4_u8(dst, v_dst);
                }
                else
                {
                    uint8x8x3_t v_dst;
                    v_dst.val[0] = v_dst0;
                    v_dst.val[1] = v_dst1;
                    v_dst.val[2] = v_dst2;
                    vst3_u8(dst, v_dst);
                }
            }
            #endif

            for( ; j < dn*3; j += 3, dst += dcn )
            {
                dst[0] = saturate_cast<uchar>(buf[j]*255.f);
                dst[1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[2] = saturate_cast<uchar>(buf[j+2]*255.f);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }
    }

    int dstcn;
    HSV2RGB_f cvt;
    #if CV_NEON
    float32x4_t v_scale, v_scale_inv;
    uint8x8_t v_alpha;
    #endif
};


///////////////////////////////////// RGB <-> HLS ////////////////////////////////////////

struct RGB2HLS_f
{
    typedef float channel_type;

    RGB2HLS_f(int _srccn, int _blueIdx, float _hrange)
    : srccn(_srccn), blueIdx(_blueIdx), hscale(_hrange/360.f) {
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int i = 0, bidx = blueIdx, scn = srccn;
        n *= 3;

        for( ; i < n; i += 3, src += scn )
        {
            float b = src[bidx], g = src[1], r = src[bidx^2];
            float h = 0.f, s = 0.f, l;
            float vmin, vmax, diff;

            vmax = vmin = r;
            if( vmax < g ) vmax = g;
            if( vmax < b ) vmax = b;
            if( vmin > g ) vmin = g;
            if( vmin > b ) vmin = b;

            diff = vmax - vmin;
            l = (vmax + vmin)*0.5f;

            if( diff > FLT_EPSILON )
            {
                s = l < 0.5f ? diff/(vmax + vmin) : diff/(2 - vmax - vmin);
                diff = 60.f/diff;

                if( vmax == r )
                    h = (g - b)*diff;
                else if( vmax == g )
                    h = (b - r)*diff + 120.f;
                else
                    h = (r - g)*diff + 240.f;

                if( h < 0.f ) h += 360.f;
            }

            dst[i] = h*hscale;
            dst[i+1] = l;
            dst[i+2] = s;
        }
    }

    int srccn, blueIdx;
    float hscale;
};


struct RGB2HLS_b
{
    typedef uchar channel_type;

    RGB2HLS_b(int _srccn, int _blueIdx, int _hrange)
    : srccn(_srccn), cvt(3, _blueIdx, (float)_hrange)
    {
        #if CV_NEON
        v_scale_inv = vdupq_n_f32(1.f/255.f);
        v_scale = vdupq_n_f32(255.f);
        v_alpha = vdup_n_u8(ColorChannel<uchar>::max());
        #endif
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, scn = srccn;
        float CV_DECL_ALIGNED(16) buf[3*BLOCK_SIZE];

        for( i = 0; i < n; i += BLOCK_SIZE, dst += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);
            j = 0;

            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24, src += 8 * scn)
            {
                uint16x8_t v_t0, v_t1, v_t2;

                if (scn == 3)
                {
                    uint8x8x3_t v_src = vld3_u8(src);
                    v_t0 = vmovl_u8(v_src.val[0]);
                    v_t1 = vmovl_u8(v_src.val[1]);
                    v_t2 = vmovl_u8(v_src.val[2]);
                }
                else
                {
                    uint8x8x4_t v_src = vld4_u8(src);
                    v_t0 = vmovl_u8(v_src.val[0]);
                    v_t1 = vmovl_u8(v_src.val[1]);
                    v_t2 = vmovl_u8(v_src.val[2]);
                }

                float32x4x3_t v_dst;
                v_dst.val[0] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t0))), v_scale_inv);
                v_dst.val[1] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t1))), v_scale_inv);
                v_dst.val[2] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t2))), v_scale_inv);
                vst3q_f32(buf + j, v_dst);

                v_dst.val[0] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t0))), v_scale_inv);
                v_dst.val[1] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t1))), v_scale_inv);
                v_dst.val[2] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t2))), v_scale_inv);
                vst3q_f32(buf + j + 12, v_dst);
            }
            #endif
            for( ; j < dn*3; j += 3, src += scn )
            {
                buf[j] = src[0]*(1.f/255.f);
                buf[j+1] = src[1]*(1.f/255.f);
                buf[j+2] = src[2]*(1.f/255.f);
            }
            cvt(buf, buf, dn);

            j = 0;
            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24)
            {
                float32x4x3_t v_src0 = vld3q_f32(buf + j), v_src1 = vld3q_f32(buf + j + 12);

                uint8x8x3_t v_dst;
                v_dst.val[0] = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_src0.val[0])),
                                                       vqmovn_u32(cv_vrndq_u32_f32(v_src1.val[0]))));
                v_dst.val[1] = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[1], v_scale))),
                                                       vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[1], v_scale)))));
                v_dst.val[2] = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[2], v_scale))),
                                                       vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[2], v_scale)))));
                vst3_u8(dst + j, v_dst);
            }
            #endif
            for( ; j < dn*3; j += 3 )
            {
                dst[j] = saturate_cast<uchar>(buf[j]);
                dst[j+1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[j+2] = saturate_cast<uchar>(buf[j+2]*255.f);
            }
        }
    }

    int srccn;
    RGB2HLS_f cvt;
    #if CV_NEON
    float32x4_t v_scale, v_scale_inv;
    uint8x8_t v_alpha;
    #endif
};


struct HLS2RGB_f
{
    typedef float channel_type;

    HLS2RGB_f(int _dstcn, int _blueIdx, float _hrange)
    : dstcn(_dstcn), blueIdx(_blueIdx), hscale(6.f/_hrange) {
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int i = 0, bidx = blueIdx, dcn = dstcn;
        float _hscale = hscale;
        float alpha = ColorChannel<float>::max();
        n *= 3;

        for( ; i < n; i += 3, dst += dcn )
        {
            float h = src[i], l = src[i+1], s = src[i+2];
            float b, g, r;

            if( s == 0 )
                b = g = r = l;
            else
            {
                static const int sector_data[][3]=
                {{1,3,0}, {1,0,2}, {3,0,1}, {0,2,1}, {0,1,3}, {2,1,0}};
                float tab[4];
                int sector;

                float p2 = l <= 0.5f ? l*(1 + s) : l + s - l*s;
                float p1 = 2*l - p2;

                h *= _hscale;
                if( h < 0 )
                    do h += 6; while( h < 0 );
                else if( h >= 6 )
                    do h -= 6; while( h >= 6 );

                assert( 0 <= h && h < 6 );
                sector = cvFloor(h);
                h -= sector;

                tab[0] = p2;
                tab[1] = p1;
                tab[2] = p1 + (p2 - p1)*(1-h);
                tab[3] = p1 + (p2 - p1)*h;

                b = tab[sector_data[sector][0]];
                g = tab[sector_data[sector][1]];
                r = tab[sector_data[sector][2]];
            }

            dst[bidx] = b;
            dst[1] = g;
            dst[bidx^2] = r;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn, blueIdx;
    float hscale;
};


struct HLS2RGB_b
{
    typedef uchar channel_type;

    HLS2RGB_b(int _dstcn, int _blueIdx, int _hrange)
    : dstcn(_dstcn), cvt(3, _blueIdx, (float)_hrange)
    {
        #if CV_NEON
        v_scale_inv = vdupq_n_f32(1.f/255.f);
        v_scale = vdupq_n_f32(255.f);
        v_alpha = vdup_n_u8(ColorChannel<uchar>::max());
        #endif
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();
        float CV_DECL_ALIGNED(16) buf[3*BLOCK_SIZE];

        for( i = 0; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);
            j = 0;

            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24)
            {
                uint8x8x3_t v_src = vld3_u8(src + j);
                uint16x8_t v_t0 = vmovl_u8(v_src.val[0]),
                           v_t1 = vmovl_u8(v_src.val[1]),
                           v_t2 = vmovl_u8(v_src.val[2]);

                float32x4x3_t v_dst;
                v_dst.val[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t0)));
                v_dst.val[1] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t1))), v_scale_inv);
                v_dst.val[2] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t2))), v_scale_inv);
                vst3q_f32(buf + j, v_dst);

                v_dst.val[0] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t0)));
                v_dst.val[1] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t1))), v_scale_inv);
                v_dst.val[2] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t2))), v_scale_inv);
                vst3q_f32(buf + j + 12, v_dst);
            }
            #endif
            for( ; j < dn*3; j += 3 )
            {
                buf[j] = src[j];
                buf[j+1] = src[j+1]*(1.f/255.f);
                buf[j+2] = src[j+2]*(1.f/255.f);
            }
            cvt(buf, buf, dn);

            j = 0;
            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24, dst += dcn * 8)
            {
                float32x4x3_t v_src0 = vld3q_f32(buf + j), v_src1 = vld3q_f32(buf + j + 12);
                uint8x8_t v_dst0 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[0], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[0], v_scale)))));
                uint8x8_t v_dst1 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[1], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[1], v_scale)))));
                uint8x8_t v_dst2 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[2], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[2], v_scale)))));

                if (dcn == 4)
                {
                    uint8x8x4_t v_dst;
                    v_dst.val[0] = v_dst0;
                    v_dst.val[1] = v_dst1;
                    v_dst.val[2] = v_dst2;
                    v_dst.val[3] = v_alpha;
                    vst4_u8(dst, v_dst);
                }
                else
                {
                    uint8x8x3_t v_dst;
                    v_dst.val[0] = v_dst0;
                    v_dst.val[1] = v_dst1;
                    v_dst.val[2] = v_dst2;
                    vst3_u8(dst, v_dst);
                }
            }
            #endif

            for( ; j < dn*3; j += 3, dst += dcn )
            {
                dst[0] = saturate_cast<uchar>(buf[j]*255.f);
                dst[1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[2] = saturate_cast<uchar>(buf[j+2]*255.f);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }
    }

    int dstcn;
    HLS2RGB_f cvt;
    #if CV_NEON
    float32x4_t v_scale, v_scale_inv;
    uint8x8_t v_alpha;
    #endif
};


///////////////////////////////////// RGB <-> L*a*b* /////////////////////////////////////

static const float D65[] = { 0.950456f, 1.f, 1.088754f };

enum { LAB_CBRT_TAB_SIZE = 1024, GAMMA_TAB_SIZE = 1024 };
static float LabCbrtTab[LAB_CBRT_TAB_SIZE*4];
static const float LabCbrtTabScale = LAB_CBRT_TAB_SIZE/1.5f;

static float sRGBGammaTab[GAMMA_TAB_SIZE*4], sRGBInvGammaTab[GAMMA_TAB_SIZE*4];
static const float GammaTabScale = (float)GAMMA_TAB_SIZE;

static ushort sRGBGammaTab_b[256], linearGammaTab_b[256];
#undef lab_shift
#define lab_shift xyz_shift
#define gamma_shift 3
#define lab_shift2 (lab_shift + gamma_shift)
#define LAB_CBRT_TAB_SIZE_B (256*3/2*(1<<gamma_shift))
static ushort LabCbrtTab_b[LAB_CBRT_TAB_SIZE_B];

static void initLabTabs()
{
    static bool initialized = false;
    if(!initialized)
    {
        float f[LAB_CBRT_TAB_SIZE+1], g[GAMMA_TAB_SIZE+1], ig[GAMMA_TAB_SIZE+1], scale = 1.f/LabCbrtTabScale;
        int i;
        for(i = 0; i <= LAB_CBRT_TAB_SIZE; i++)
        {
            float x = i*scale;
            f[i] = x < 0.008856f ? x*7.787f + 0.13793103448275862f : cvCbrt(x);
        }
        splineBuild(f, LAB_CBRT_TAB_SIZE, LabCbrtTab);

        scale = 1.f/GammaTabScale;
        for(i = 0; i <= GAMMA_TAB_SIZE; i++)
        {
            float x = i*scale;
            g[i] = x <= 0.04045f ? x*(1.f/12.92f) : (float)std::pow((double)(x + 0.055)*(1./1.055), 2.4);
            ig[i] = x <= 0.0031308 ? x*12.92f : (float)(1.055*std::pow((double)x, 1./2.4) - 0.055);
        }
        splineBuild(g, GAMMA_TAB_SIZE, sRGBGammaTab);
        splineBuild(ig, GAMMA_TAB_SIZE, sRGBInvGammaTab);

        for(i = 0; i < 256; i++)
        {
            float x = i*(1.f/255.f);
            sRGBGammaTab_b[i] = saturate_cast<ushort>(255.f*(1 << gamma_shift)*(x <= 0.04045f ? x*(1.f/12.92f) : (float)std::pow((double)(x + 0.055)*(1./1.055), 2.4)));
            linearGammaTab_b[i] = (ushort)(i*(1 << gamma_shift));
        }

        for(i = 0; i < LAB_CBRT_TAB_SIZE_B; i++)
        {
            float x = i*(1.f/(255.f*(1 << gamma_shift)));
            LabCbrtTab_b[i] = saturate_cast<ushort>((1 << lab_shift2)*(x < 0.008856f ? x*7.787f + 0.13793103448275862f : cvCbrt(x)));
        }
        initialized = true;
    }
}

struct RGB2Lab_b
{
    typedef uchar channel_type;

    RGB2Lab_b(int _srccn, int blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb)
    : srccn(_srccn), srgb(_srgb)
    {
        static volatile int _3 = 3;
        initLabTabs();

        if (!_coeffs)
            _coeffs = sRGB2XYZ_D65;
        if (!_whitept)
            _whitept = D65;

        float scale[] =
        {
            (1 << lab_shift)/_whitept[0],
            (float)(1 << lab_shift),
            (1 << lab_shift)/_whitept[2]
        };

        for( int i = 0; i < _3; i++ )
        {
            coeffs[i*3+(blueIdx^2)] = cvRound(_coeffs[i*3]*scale[i]);
            coeffs[i*3+1] = cvRound(_coeffs[i*3+1]*scale[i]);
            coeffs[i*3+blueIdx] = cvRound(_coeffs[i*3+2]*scale[i]);

            CV_Assert( coeffs[i] >= 0 && coeffs[i*3+1] >= 0 && coeffs[i*3+2] >= 0 &&
                      coeffs[i*3] + coeffs[i*3+1] + coeffs[i*3+2] < 2*(1 << lab_shift) );
        }
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        const int Lscale = (116*255+50)/100;
        const int Lshift = -((16*255*(1 << lab_shift2) + 50)/100);
        const ushort* tab = srgb ? sRGBGammaTab_b : linearGammaTab_b;
        int i, scn = srccn;
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;

        for( i = 0; i < n; i += 3, src += scn )
        {
            int R = tab[src[0]], G = tab[src[1]], B = tab[src[2]];
            int fX = LabCbrtTab_b[CV_DESCALE(R*C0 + G*C1 + B*C2, lab_shift)];
            int fY = LabCbrtTab_b[CV_DESCALE(R*C3 + G*C4 + B*C5, lab_shift)];
            int fZ = LabCbrtTab_b[CV_DESCALE(R*C6 + G*C7 + B*C8, lab_shift)];

            int L = CV_DESCALE( Lscale*fY + Lshift, lab_shift2 );
            int a = CV_DESCALE( 500*(fX - fY) + 128*(1 << lab_shift2), lab_shift2 );
            int b = CV_DESCALE( 200*(fY - fZ) + 128*(1 << lab_shift2), lab_shift2 );

            dst[i] = saturate_cast<uchar>(L);
            dst[i+1] = saturate_cast<uchar>(a);
            dst[i+2] = saturate_cast<uchar>(b);
        }
    }

    int srccn;
    int coeffs[9];
    bool srgb;
};


#define clip(value) \
    value < 0.0f ? 0.0f : value > 1.0f ? 1.0f : value;

struct RGB2Lab_f
{
    typedef float channel_type;

    RGB2Lab_f(int _srccn, int blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb)
    : srccn(_srccn), srgb(_srgb)
    {
        volatile int _3 = 3;
        initLabTabs();

        if (!_coeffs)
            _coeffs = sRGB2XYZ_D65;
        if (!_whitept)
            _whitept = D65;

        float scale[] = { 1.0f / _whitept[0], 1.0f, 1.0f / _whitept[2] };

        for( int i = 0; i < _3; i++ )
        {
            int j = i * 3;
            coeffs[j + (blueIdx ^ 2)] = _coeffs[j] * scale[i];
            coeffs[j + 1] = _coeffs[j + 1] * scale[i];
            coeffs[j + blueIdx] = _coeffs[j + 2] * scale[i];

            CV_Assert( coeffs[j] >= 0 && coeffs[j + 1] >= 0 && coeffs[j + 2] >= 0 &&
                       coeffs[j] + coeffs[j + 1] + coeffs[j + 2] < 1.5f*LabCbrtTabScale );
        }
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int i, scn = srccn;
        float gscale = GammaTabScale;
        const float* gammaTab = srgb ? sRGBGammaTab : 0;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;

        static const float _1_3 = 1.0f / 3.0f;
        static const float _a = 16.0f / 116.0f;
        for (i = 0; i < n; i += 3, src += scn )
        {
            float R = clip(src[0]);
            float G = clip(src[1]);
            float B = clip(src[2]);

            if (gammaTab)
            {
                R = splineInterpolate(R * gscale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G * gscale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B * gscale, gammaTab, GAMMA_TAB_SIZE);
            }
            float X = R*C0 + G*C1 + B*C2;
            float Y = R*C3 + G*C4 + B*C5;
            float Z = R*C6 + G*C7 + B*C8;

            float FX = X > 0.008856f ? std::pow(X, _1_3) : (7.787f * X + _a);
            float FY = Y > 0.008856f ? std::pow(Y, _1_3) : (7.787f * Y + _a);
            float FZ = Z > 0.008856f ? std::pow(Z, _1_3) : (7.787f * Z + _a);

            float L = Y > 0.008856f ? (116.f * FY - 16.f) : (903.3f * Y);
            float a = 500.f * (FX - FY);
            float b = 200.f * (FY - FZ);

            dst[i] = L;
            dst[i + 1] = a;
            dst[i + 2] = b;
        }
    }

    int srccn;
    float coeffs[9];
    bool srgb;
};

struct Lab2RGB_f
{
    typedef float channel_type;

    Lab2RGB_f( int _dstcn, int blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb )
    : dstcn(_dstcn), srgb(_srgb)
    {
        initLabTabs();

        if(!_coeffs)
            _coeffs = XYZ2sRGB_D65;
        if(!_whitept)
            _whitept = D65;

        for( int i = 0; i < 3; i++ )
        {
            coeffs[i+(blueIdx^2)*3] = _coeffs[i]*_whitept[i];
            coeffs[i+3] = _coeffs[i+3]*_whitept[i];
            coeffs[i+blueIdx*3] = _coeffs[i+6]*_whitept[i];
        }

        lThresh = 0.008856f * 903.3f;
        fThresh = 7.787f * 0.008856f + 16.0f / 116.0f;
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int i = 0, dcn = dstcn;
        const float* gammaTab = srgb ? sRGBInvGammaTab : 0;
        float gscale = GammaTabScale;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
        C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
        C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        float alpha = ColorChannel<float>::max();
        n *= 3;

        for (; i < n; i += 3, dst += dcn)
        {
            float li = src[i];
            float ai = src[i + 1];
            float bi = src[i + 2];

            float y, fy;
            if (li <= lThresh)
            {
                y = li / 903.3f;
                fy = 7.787f * y + 16.0f / 116.0f;
            }
            else
            {
                fy = (li + 16.0f) / 116.0f;
                y = fy * fy * fy;
            }

            float fxz[] = { ai / 500.0f + fy, fy - bi / 200.0f };

            for (int j = 0; j < 2; j++)
                if (fxz[j] <= fThresh)
                    fxz[j] = (fxz[j] - 16.0f / 116.0f) / 7.787f;
                else
                    fxz[j] = fxz[j] * fxz[j] * fxz[j];


            float x = fxz[0], z = fxz[1];
            float ro = C0 * x + C1 * y + C2 * z;
            float go = C3 * x + C4 * y + C5 * z;
            float bo = C6 * x + C7 * y + C8 * z;
            ro = clip(ro);
            go = clip(go);
            bo = clip(bo);

            if (gammaTab)
            {
                ro = splineInterpolate(ro * gscale, gammaTab, GAMMA_TAB_SIZE);
                go = splineInterpolate(go * gscale, gammaTab, GAMMA_TAB_SIZE);
                bo = splineInterpolate(bo * gscale, gammaTab, GAMMA_TAB_SIZE);
            }

            dst[0] = ro, dst[1] = go, dst[2] = bo;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn;
    float coeffs[9];
    bool srgb;
    float lThresh;
    float fThresh;
};

#undef clip

struct Lab2RGB_b
{
    typedef uchar channel_type;

    Lab2RGB_b( int _dstcn, int blueIdx, const float* _coeffs,
               const float* _whitept, bool _srgb )
    : dstcn(_dstcn), cvt(3, blueIdx, _coeffs, _whitept, _srgb )
    {
        #if CV_NEON
        v_scale_inv = vdupq_n_f32(100.f/255.f);
        v_scale = vdupq_n_f32(255.f);
        v_alpha = vdup_n_u8(ColorChannel<uchar>::max());
        v_128 = vdupq_n_f32(128.0f);
        #endif
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();
        float CV_DECL_ALIGNED(16) buf[3*BLOCK_SIZE];

        for( i = 0; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);
            j = 0;

            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24)
            {
                uint8x8x3_t v_src = vld3_u8(src + j);
                uint16x8_t v_t0 = vmovl_u8(v_src.val[0]),
                           v_t1 = vmovl_u8(v_src.val[1]),
                           v_t2 = vmovl_u8(v_src.val[2]);

                float32x4x3_t v_dst;
                v_dst.val[0] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t0))), v_scale_inv);
                v_dst.val[1] = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t1))), v_128);
                v_dst.val[2] = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t2))), v_128);
                vst3q_f32(buf + j, v_dst);

                v_dst.val[0] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t0))), v_scale_inv);
                v_dst.val[1] = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t1))), v_128);
                v_dst.val[2] = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t2))), v_128);
                vst3q_f32(buf + j + 12, v_dst);
            }
            #endif

            for( ; j < dn*3; j += 3 )
            {
                buf[j] = src[j]*(100.f/255.f);
                buf[j+1] = (float)(src[j+1] - 128);
                buf[j+2] = (float)(src[j+2] - 128);
            }
            cvt(buf, buf, dn);
            j = 0;

            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24, dst += dcn * 8)
            {
                float32x4x3_t v_src0 = vld3q_f32(buf + j), v_src1 = vld3q_f32(buf + j + 12);
                uint8x8_t v_dst0 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[0], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[0], v_scale)))));
                uint8x8_t v_dst1 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[1], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[1], v_scale)))));
                uint8x8_t v_dst2 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[2], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[2], v_scale)))));

                if (dcn == 4)
                {
                    uint8x8x4_t v_dst;
                    v_dst.val[0] = v_dst0;
                    v_dst.val[1] = v_dst1;
                    v_dst.val[2] = v_dst2;
                    v_dst.val[3] = v_alpha;
                    vst4_u8(dst, v_dst);
                }
                else
                {
                    uint8x8x3_t v_dst;
                    v_dst.val[0] = v_dst0;
                    v_dst.val[1] = v_dst1;
                    v_dst.val[2] = v_dst2;
                    vst3_u8(dst, v_dst);
                }
            }
            #endif

            for( ; j < dn*3; j += 3, dst += dcn )
            {
                dst[0] = saturate_cast<uchar>(buf[j]*255.f);
                dst[1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[2] = saturate_cast<uchar>(buf[j+2]*255.f);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }
    }

    int dstcn;
    Lab2RGB_f cvt;

    #if CV_NEON
    float32x4_t v_scale, v_scale_inv, v_128;
    uint8x8_t v_alpha;
    #endif
};


///////////////////////////////////// RGB <-> L*u*v* /////////////////////////////////////

struct RGB2Luv_f
{
    typedef float channel_type;

    RGB2Luv_f( int _srccn, int blueIdx, const float* _coeffs,
               const float* whitept, bool _srgb )
    : srccn(_srccn), srgb(_srgb)
    {
        volatile int i;
        initLabTabs();

        if(!_coeffs) _coeffs = sRGB2XYZ_D65;
        if(!whitept) whitept = D65;

        for( i = 0; i < 3; i++ )
        {
            coeffs[i*3] = _coeffs[i*3];
            coeffs[i*3+1] = _coeffs[i*3+1];
            coeffs[i*3+2] = _coeffs[i*3+2];
            if( blueIdx == 0 )
                std::swap(coeffs[i*3], coeffs[i*3+2]);
            CV_Assert( coeffs[i*3] >= 0 && coeffs[i*3+1] >= 0 && coeffs[i*3+2] >= 0 &&
                      coeffs[i*3] + coeffs[i*3+1] + coeffs[i*3+2] < 1.5f );
        }

        float d = 1.f/(whitept[0] + whitept[1]*15 + whitept[2]*3);
        un = 4*whitept[0]*d*13;
        vn = 9*whitept[1]*d*13;

        CV_Assert(whitept[1] == 1.f);
    }

    #if CV_NEON
    void process(float32x4x3_t& v_src) const
    {
        float32x4_t v_x = vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], vdupq_n_f32(coeffs[0])), v_src.val[1], vdupq_n_f32(coeffs[1])), v_src.val[2], vdupq_n_f32(coeffs[2]));
        float32x4_t v_y = vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], vdupq_n_f32(coeffs[3])), v_src.val[1], vdupq_n_f32(coeffs[4])), v_src.val[2], vdupq_n_f32(coeffs[5]));
        float32x4_t v_z = vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], vdupq_n_f32(coeffs[6])), v_src.val[1], vdupq_n_f32(coeffs[7])), v_src.val[2], vdupq_n_f32(coeffs[8]));

        v_src.val[0] = vmulq_f32(v_y, vdupq_n_f32(LabCbrtTabScale));
        splineInterpolate(v_src.val[0], LabCbrtTab, LAB_CBRT_TAB_SIZE);

        v_src.val[0] = vmlaq_f32(vdupq_n_f32(-16.f), v_src.val[0], vdupq_n_f32(116.f));

        float32x4_t v_div = vmaxq_f32(vmlaq_f32(vmlaq_f32(v_x, vdupq_n_f32(15.f), v_y), vdupq_n_f32(3.f), v_z), vdupq_n_f32(FLT_EPSILON));
        float32x4_t v_reciprocal = vrecpeq_f32(v_div);
        v_reciprocal = vmulq_f32(vrecpsq_f32(v_div, v_reciprocal), v_reciprocal);
        v_reciprocal = vmulq_f32(vrecpsq_f32(v_div, v_reciprocal), v_reciprocal);
        float32x4_t v_d = vmulq_f32(vdupq_n_f32(52.f), v_reciprocal);

        v_src.val[1] = vmulq_f32(v_src.val[0], vmlaq_f32(vdupq_n_f32(-un), v_x, v_d));
        v_src.val[2] = vmulq_f32(v_src.val[0], vmlaq_f32(vdupq_n_f32(-vn), vmulq_f32(vdupq_n_f32(2.25f), v_y), v_d));
    }
    #endif

    void operator()(const float* src, float* dst, int n) const
    {
        int i = 0, scn = srccn;
        float gscale = GammaTabScale;
        const float* gammaTab = srgb ? sRGBGammaTab : 0;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;

        #if CV_NEON
        if (scn == 3)
        {
            for( ; i <= n - 12; i += 12, src += scn * 4 )
            {
                float32x4x3_t v_src = vld3q_f32(src);
                if( gammaTab )
                {
                    v_src.val[0] = vmulq_f32(v_src.val[0], vdupq_n_f32(gscale));
                    v_src.val[1] = vmulq_f32(v_src.val[1], vdupq_n_f32(gscale));
                    v_src.val[2] = vmulq_f32(v_src.val[2], vdupq_n_f32(gscale));
                    splineInterpolate(v_src.val[0], gammaTab, GAMMA_TAB_SIZE);
                    splineInterpolate(v_src.val[1], gammaTab, GAMMA_TAB_SIZE);
                    splineInterpolate(v_src.val[2], gammaTab, GAMMA_TAB_SIZE);
                }

                process(v_src);

                vst3q_f32(dst + i, v_src);
            }
        }
        else
        {
            for( ; i <= n - 12; i += 12, src += scn * 4 )
            {
                float32x4x4_t v_src = vld4q_f32(src);
                if( gammaTab )
                {
                    v_src.val[0] = vmulq_f32(v_src.val[0], vdupq_n_f32(gscale));
                    v_src.val[1] = vmulq_f32(v_src.val[1], vdupq_n_f32(gscale));
                    v_src.val[2] = vmulq_f32(v_src.val[2], vdupq_n_f32(gscale));
                    splineInterpolate(v_src.val[0], gammaTab, GAMMA_TAB_SIZE);
                    splineInterpolate(v_src.val[1], gammaTab, GAMMA_TAB_SIZE);
                    splineInterpolate(v_src.val[2], gammaTab, GAMMA_TAB_SIZE);
                }

                float32x4x3_t v_dst;
                v_dst.val[0] = v_src.val[0];
                v_dst.val[1] = v_src.val[1];
                v_dst.val[2] = v_src.val[2];
                process(v_dst);

                vst3q_f32(dst + i, v_dst);
            }
        }
        #endif
        for( ; i < n; i += 3, src += scn )
        {
            float R = src[0], G = src[1], B = src[2];
            if( gammaTab )
            {
                R = splineInterpolate(R*gscale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G*gscale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B*gscale, gammaTab, GAMMA_TAB_SIZE);
            }

            float X = R*C0 + G*C1 + B*C2;
            float Y = R*C3 + G*C4 + B*C5;
            float Z = R*C6 + G*C7 + B*C8;

            float L = splineInterpolate(Y*LabCbrtTabScale, LabCbrtTab, LAB_CBRT_TAB_SIZE);
            L = 116.f*L - 16.f;

            float d = (4*13) / std::max(X + 15 * Y + 3 * Z, FLT_EPSILON);
            float u = L*(X*d - un);
            float v = L*((9*0.25f)*Y*d - vn);

            dst[i] = L; dst[i+1] = u; dst[i+2] = v;
        }
    }

    int srccn;
    float coeffs[9], un, vn;
    bool srgb;
};


struct Luv2RGB_f
{
    typedef float channel_type;

    Luv2RGB_f( int _dstcn, int blueIdx, const float* _coeffs,
              const float* whitept, bool _srgb )
    : dstcn(_dstcn), srgb(_srgb)
    {
        initLabTabs();

        if(!_coeffs) _coeffs = XYZ2sRGB_D65;
        if(!whitept) whitept = D65;

        for( int i = 0; i < 3; i++ )
        {
            coeffs[i+(blueIdx^2)*3] = _coeffs[i];
            coeffs[i+3] = _coeffs[i+3];
            coeffs[i+blueIdx*3] = _coeffs[i+6];
        }

        float d = 1.f/(whitept[0] + whitept[1]*15 + whitept[2]*3);
        un = 4*whitept[0]*d;
        vn = 9*whitept[1]*d;

        CV_Assert(whitept[1] == 1.f);
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int i = 0, dcn = dstcn;
        const float* gammaTab = srgb ? sRGBInvGammaTab : 0;
        float gscale = GammaTabScale;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        float alpha = ColorChannel<float>::max();
        float _un = un, _vn = vn;
        n *= 3;

        for( ; i < n; i += 3, dst += dcn )
        {
            float L = src[i], u = src[i+1], v = src[i+2], d, X, Y, Z;
            Y = (L + 16.f) * (1.f/116.f);
            Y = Y*Y*Y;
            d = (1.f/13.f)/L;
            u = u*d + _un;
            v = v*d + _vn;
            float iv = 1.f/v;
            X = 2.25f * u * Y * iv ;
            Z = (12 - 3 * u - 20 * v) * Y * 0.25f * iv;

            float R = X*C0 + Y*C1 + Z*C2;
            float G = X*C3 + Y*C4 + Z*C5;
            float B = X*C6 + Y*C7 + Z*C8;

            R = std::min(std::max(R, 0.f), 1.f);
            G = std::min(std::max(G, 0.f), 1.f);
            B = std::min(std::max(B, 0.f), 1.f);

            if( gammaTab )
            {
                R = splineInterpolate(R*gscale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G*gscale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B*gscale, gammaTab, GAMMA_TAB_SIZE);
            }

            dst[0] = R; dst[1] = G; dst[2] = B;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn;
    float coeffs[9], un, vn;
    bool srgb;
};


struct RGB2Luv_b
{
    typedef uchar channel_type;

    RGB2Luv_b( int _srccn, int blueIdx, const float* _coeffs,
               const float* _whitept, bool _srgb )
    : srccn(_srccn), cvt(3, blueIdx, _coeffs, _whitept, _srgb)
    {
        #if CV_NEON
        v_scale_inv = vdupq_n_f32(1.f/255.f);
        v_scale = vdupq_n_f32(2.55f);
        v_coeff1 = vdupq_n_f32(0.72033898305084743f);
        v_coeff2 = vdupq_n_f32(96.525423728813564f);
        v_coeff3 = vdupq_n_f32(0.9732824427480916f);
        v_coeff4 = vdupq_n_f32(136.259541984732824f);
        v_alpha = vdup_n_u8(ColorChannel<uchar>::max());
        #endif
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, scn = srccn;
        float CV_DECL_ALIGNED(16) buf[3*BLOCK_SIZE];

        for( i = 0; i < n; i += BLOCK_SIZE, dst += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);
            j = 0;

            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24, src += 8 * scn)
            {
                uint16x8_t v_t0, v_t1, v_t2;

                if (scn == 3)
                {
                    uint8x8x3_t v_src = vld3_u8(src);
                    v_t0 = vmovl_u8(v_src.val[0]);
                    v_t1 = vmovl_u8(v_src.val[1]);
                    v_t2 = vmovl_u8(v_src.val[2]);
                }
                else
                {
                    uint8x8x4_t v_src = vld4_u8(src);
                    v_t0 = vmovl_u8(v_src.val[0]);
                    v_t1 = vmovl_u8(v_src.val[1]);
                    v_t2 = vmovl_u8(v_src.val[2]);
                }

                float32x4x3_t v_dst;
                v_dst.val[0] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t0))), v_scale_inv);
                v_dst.val[1] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t1))), v_scale_inv);
                v_dst.val[2] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t2))), v_scale_inv);
                vst3q_f32(buf + j, v_dst);

                v_dst.val[0] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t0))), v_scale_inv);
                v_dst.val[1] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t1))), v_scale_inv);
                v_dst.val[2] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t2))), v_scale_inv);
                vst3q_f32(buf + j + 12, v_dst);
            }
            #endif
            for( ; j < dn*3; j += 3, src += scn )
            {
                buf[j] = src[0]*(1.f/255.f);
                buf[j+1] = (float)(src[1]*(1.f/255.f));
                buf[j+2] = (float)(src[2]*(1.f/255.f));
            }
            cvt(buf, buf, dn);

            j = 0;
            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24)
            {
                float32x4x3_t v_src0 = vld3q_f32(buf + j), v_src1 = vld3q_f32(buf + j + 12);

                uint8x8x3_t v_dst;
                v_dst.val[0] = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[0], v_scale))),
                                                       vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[0], v_scale)))));
                v_dst.val[1] = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vaddq_f32(vmulq_f32(v_src0.val[1], v_coeff1), v_coeff2))),
                                                       vqmovn_u32(cv_vrndq_u32_f32(vaddq_f32(vmulq_f32(v_src1.val[1], v_coeff1), v_coeff2)))));
                v_dst.val[2] = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vaddq_f32(vmulq_f32(v_src0.val[2], v_coeff3), v_coeff4))),
                                                       vqmovn_u32(cv_vrndq_u32_f32(vaddq_f32(vmulq_f32(v_src1.val[2], v_coeff3), v_coeff4)))));

                vst3_u8(dst + j, v_dst);
            }
            #endif

            for( ; j < dn*3; j += 3 )
            {
                dst[j] = saturate_cast<uchar>(buf[j]*2.55f);
                dst[j+1] = saturate_cast<uchar>(buf[j+1]*0.72033898305084743f + 96.525423728813564f);
                dst[j+2] = saturate_cast<uchar>(buf[j+2]*0.9732824427480916f + 136.259541984732824f);
            }
        }
    }

    int srccn;
    RGB2Luv_f cvt;

    #if CV_NEON
    float32x4_t v_scale, v_scale_inv, v_coeff1, v_coeff2, v_coeff3, v_coeff4;
    uint8x8_t v_alpha;
    #endif
};


struct Luv2RGB_b
{
    typedef uchar channel_type;

    Luv2RGB_b( int _dstcn, int blueIdx, const float* _coeffs,
               const float* _whitept, bool _srgb )
    : dstcn(_dstcn), cvt(3, blueIdx, _coeffs, _whitept, _srgb )
    {
        #if CV_NEON
        v_scale_inv = vdupq_n_f32(100.f/255.f);
        v_coeff1 = vdupq_n_f32(1.388235294117647f);
        v_coeff2 = vdupq_n_f32(1.027450980392157f);
        v_134 = vdupq_n_f32(134.f);
        v_140 = vdupq_n_f32(140.f);
        v_scale = vdupq_n_f32(255.f);
        v_alpha = vdup_n_u8(ColorChannel<uchar>::max());
        #endif
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();
        float CV_DECL_ALIGNED(16) buf[3*BLOCK_SIZE];

        for( i = 0; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);
            j = 0;

            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24)
            {
                uint8x8x3_t v_src = vld3_u8(src + j);
                uint16x8_t v_t0 = vmovl_u8(v_src.val[0]),
                           v_t1 = vmovl_u8(v_src.val[1]),
                           v_t2 = vmovl_u8(v_src.val[2]);

                float32x4x3_t v_dst;
                v_dst.val[0] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t0))), v_scale_inv);
                v_dst.val[1] = vsubq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t1))), v_coeff1), v_134);
                v_dst.val[2] = vsubq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t2))), v_coeff2), v_140);
                vst3q_f32(buf + j, v_dst);

                v_dst.val[0] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t0))), v_scale_inv);
                v_dst.val[1] = vsubq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t1))), v_coeff1), v_134);
                v_dst.val[2] = vsubq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t2))), v_coeff2), v_140);
                vst3q_f32(buf + j + 12, v_dst);
            }
            #endif
            for( ; j < dn*3; j += 3 )
            {
                buf[j] = src[j]*(100.f/255.f);
                buf[j+1] = (float)(src[j+1]*1.388235294117647f - 134.f);
                buf[j+2] = (float)(src[j+2]*1.027450980392157f - 140.f);
            }
            cvt(buf, buf, dn);

            j = 0;
            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24, dst += dcn * 8)
            {
                float32x4x3_t v_src0 = vld3q_f32(buf + j), v_src1 = vld3q_f32(buf + j + 12);
                uint8x8_t v_dst0 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[0], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[0], v_scale)))));
                uint8x8_t v_dst1 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[1], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[1], v_scale)))));
                uint8x8_t v_dst2 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[2], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[2], v_scale)))));

                if (dcn == 4)
                {
                    uint8x8x4_t v_dst;
                    v_dst.val[0] = v_dst0;
                    v_dst.val[1] = v_dst1;
                    v_dst.val[2] = v_dst2;
                    v_dst.val[3] = v_alpha;
                    vst4_u8(dst, v_dst);
                }
                else
                {
                    uint8x8x3_t v_dst;
                    v_dst.val[0] = v_dst0;
                    v_dst.val[1] = v_dst1;
                    v_dst.val[2] = v_dst2;
                    vst3_u8(dst, v_dst);
                }
            }
            #endif

            for( ; j < dn*3; j += 3, dst += dcn )
            {
                dst[0] = saturate_cast<uchar>(buf[j]*255.f);
                dst[1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[2] = saturate_cast<uchar>(buf[j+2]*255.f);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }
    }

    int dstcn;
    Luv2RGB_f cvt;

    #if CV_NEON
    float32x4_t v_scale, v_scale_inv, v_coeff1, v_coeff2, v_134, v_140;
    uint8x8_t v_alpha;
    #endif
};


///////////////////////////////////// YUV420 -> RGB /////////////////////////////////////

const int ITUR_BT_601_CY = 1220542;
const int ITUR_BT_601_CUB = 2116026;
const int ITUR_BT_601_CUG = -409993;
const int ITUR_BT_601_CVG = -852492;
const int ITUR_BT_601_CVR = 1673527;
const int ITUR_BT_601_SHIFT = 20;

// Coefficients for RGB to YUV420p conversion
const int ITUR_BT_601_CRY =  269484;
const int ITUR_BT_601_CGY =  528482;
const int ITUR_BT_601_CBY =  102760;
const int ITUR_BT_601_CRU = -155188;
const int ITUR_BT_601_CGU = -305135;
const int ITUR_BT_601_CBU =  460324;
const int ITUR_BT_601_CGV = -385875;
const int ITUR_BT_601_CBV = -74448;

template<int bIdx, int uIdx>
struct YUV420sp2RGB888Invoker : ParallelLoopBody
{
    uchar * dst_data;
    size_t dst_step;
    int width;
    const uchar* my1, *muv;
    size_t stride;

    YUV420sp2RGB888Invoker(uchar * _dst_data, size_t _dst_step, int _dst_width, size_t _stride, const uchar* _y1, const uchar* _uv)
        : dst_data(_dst_data), dst_step(_dst_step), width(_dst_width), my1(_y1), muv(_uv), stride(_stride) {}

    void operator()(const Range& range) const
    {
        int rangeBegin = range.start * 2;
        int rangeEnd = range.end * 2;

        //R = 1.164(Y - 16) + 1.596(V - 128)
        //G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
        //B = 1.164(Y - 16)                  + 2.018(U - 128)

        //R = (1220542(Y - 16) + 1673527(V - 128)                  + (1 << 19)) >> 20
        //G = (1220542(Y - 16) - 852492(V - 128) - 409993(U - 128) + (1 << 19)) >> 20
        //B = (1220542(Y - 16)                  + 2116026(U - 128) + (1 << 19)) >> 20

        const uchar* y1 = my1 + rangeBegin * stride, *uv = muv + rangeBegin * stride / 2;

        for (int j = rangeBegin; j < rangeEnd; j += 2, y1 += stride * 2, uv += stride)
        {
            uchar* row1 = dst_data + dst_step * j;
            uchar* row2 = dst_data + dst_step * (j + 1);
            const uchar* y2 = y1 + stride;

            for (int i = 0; i < width; i += 2, row1 += 6, row2 += 6)
            {
                int u = int(uv[i + 0 + uIdx]) - 128;
                int v = int(uv[i + 1 - uIdx]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(y1[i]) - 16) * ITUR_BT_601_CY;
                row1[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row1[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row1[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);

                int y01 = std::max(0, int(y1[i + 1]) - 16) * ITUR_BT_601_CY;
                row1[5-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row1[4]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row1[3+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);

                int y10 = std::max(0, int(y2[i]) - 16) * ITUR_BT_601_CY;
                row2[2-bIdx] = saturate_cast<uchar>((y10 + ruv) >> ITUR_BT_601_SHIFT);
                row2[1]      = saturate_cast<uchar>((y10 + guv) >> ITUR_BT_601_SHIFT);
                row2[bIdx]   = saturate_cast<uchar>((y10 + buv) >> ITUR_BT_601_SHIFT);

                int y11 = std::max(0, int(y2[i + 1]) - 16) * ITUR_BT_601_CY;
                row2[5-bIdx] = saturate_cast<uchar>((y11 + ruv) >> ITUR_BT_601_SHIFT);
                row2[4]      = saturate_cast<uchar>((y11 + guv) >> ITUR_BT_601_SHIFT);
                row2[3+bIdx] = saturate_cast<uchar>((y11 + buv) >> ITUR_BT_601_SHIFT);
            }
        }
    }
};

template<int bIdx, int uIdx>
struct YUV420sp2RGBA8888Invoker : ParallelLoopBody
{
    uchar * dst_data;
    size_t dst_step;
    int width;
    const uchar* my1, *muv;
    size_t stride;

    YUV420sp2RGBA8888Invoker(uchar * _dst_data, size_t _dst_step, int _dst_width, size_t _stride, const uchar* _y1, const uchar* _uv)
        : dst_data(_dst_data), dst_step(_dst_step), width(_dst_width), my1(_y1), muv(_uv), stride(_stride) {}

    void operator()(const Range& range) const
    {
        int rangeBegin = range.start * 2;
        int rangeEnd = range.end * 2;

        //R = 1.164(Y - 16) + 1.596(V - 128)
        //G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
        //B = 1.164(Y - 16)                  + 2.018(U - 128)

        //R = (1220542(Y - 16) + 1673527(V - 128)                  + (1 << 19)) >> 20
        //G = (1220542(Y - 16) - 852492(V - 128) - 409993(U - 128) + (1 << 19)) >> 20
        //B = (1220542(Y - 16)                  + 2116026(U - 128) + (1 << 19)) >> 20

        const uchar* y1 = my1 + rangeBegin * stride, *uv = muv + rangeBegin * stride / 2;

        for (int j = rangeBegin; j < rangeEnd; j += 2, y1 += stride * 2, uv += stride)
        {
            uchar* row1 = dst_data + dst_step * j;
            uchar* row2 = dst_data + dst_step * (j + 1);
            const uchar* y2 = y1 + stride;

            for (int i = 0; i < width; i += 2, row1 += 8, row2 += 8)
            {
                int u = int(uv[i + 0 + uIdx]) - 128;
                int v = int(uv[i + 1 - uIdx]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(y1[i]) - 16) * ITUR_BT_601_CY;
                row1[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row1[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row1[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);
                row1[3]      = uchar(0xff);

                int y01 = std::max(0, int(y1[i + 1]) - 16) * ITUR_BT_601_CY;
                row1[6-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row1[5]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row1[4+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);
                row1[7]      = uchar(0xff);

                int y10 = std::max(0, int(y2[i]) - 16) * ITUR_BT_601_CY;
                row2[2-bIdx] = saturate_cast<uchar>((y10 + ruv) >> ITUR_BT_601_SHIFT);
                row2[1]      = saturate_cast<uchar>((y10 + guv) >> ITUR_BT_601_SHIFT);
                row2[bIdx]   = saturate_cast<uchar>((y10 + buv) >> ITUR_BT_601_SHIFT);
                row2[3]      = uchar(0xff);

                int y11 = std::max(0, int(y2[i + 1]) - 16) * ITUR_BT_601_CY;
                row2[6-bIdx] = saturate_cast<uchar>((y11 + ruv) >> ITUR_BT_601_SHIFT);
                row2[5]      = saturate_cast<uchar>((y11 + guv) >> ITUR_BT_601_SHIFT);
                row2[4+bIdx] = saturate_cast<uchar>((y11 + buv) >> ITUR_BT_601_SHIFT);
                row2[7]      = uchar(0xff);
            }
        }
    }
};

template<int bIdx>
struct YUV420p2RGB888Invoker : ParallelLoopBody
{
    uchar * dst_data;
    size_t dst_step;
    int width;
    const uchar* my1, *mu, *mv;
    size_t stride;
    int ustepIdx, vstepIdx;

    YUV420p2RGB888Invoker(uchar * _dst_data, size_t _dst_step, int _dst_width, size_t _stride, const uchar* _y1, const uchar* _u, const uchar* _v, int _ustepIdx, int _vstepIdx)
        : dst_data(_dst_data), dst_step(_dst_step), width(_dst_width), my1(_y1), mu(_u), mv(_v), stride(_stride), ustepIdx(_ustepIdx), vstepIdx(_vstepIdx) {}

    void operator()(const Range& range) const
    {
        const int rangeBegin = range.start * 2;
        const int rangeEnd = range.end * 2;

        int uvsteps[2] = {width/2, static_cast<int>(stride) - width/2};
        int usIdx = ustepIdx, vsIdx = vstepIdx;

        const uchar* y1 = my1 + rangeBegin * stride;
        const uchar* u1 = mu + (range.start / 2) * stride;
        const uchar* v1 = mv + (range.start / 2) * stride;

        if(range.start % 2 == 1)
        {
            u1 += uvsteps[(usIdx++) & 1];
            v1 += uvsteps[(vsIdx++) & 1];
        }

        for (int j = rangeBegin; j < rangeEnd; j += 2, y1 += stride * 2, u1 += uvsteps[(usIdx++) & 1], v1 += uvsteps[(vsIdx++) & 1])
        {
            uchar* row1 = dst_data + dst_step * j;
            uchar* row2 = dst_data + dst_step * (j + 1);
            const uchar* y2 = y1 + stride;

            for (int i = 0; i < width / 2; i += 1, row1 += 6, row2 += 6)
            {
                int u = int(u1[i]) - 128;
                int v = int(v1[i]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(y1[2 * i]) - 16) * ITUR_BT_601_CY;
                row1[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row1[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row1[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);

                int y01 = std::max(0, int(y1[2 * i + 1]) - 16) * ITUR_BT_601_CY;
                row1[5-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row1[4]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row1[3+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);

                int y10 = std::max(0, int(y2[2 * i]) - 16) * ITUR_BT_601_CY;
                row2[2-bIdx] = saturate_cast<uchar>((y10 + ruv) >> ITUR_BT_601_SHIFT);
                row2[1]      = saturate_cast<uchar>((y10 + guv) >> ITUR_BT_601_SHIFT);
                row2[bIdx]   = saturate_cast<uchar>((y10 + buv) >> ITUR_BT_601_SHIFT);

                int y11 = std::max(0, int(y2[2 * i + 1]) - 16) * ITUR_BT_601_CY;
                row2[5-bIdx] = saturate_cast<uchar>((y11 + ruv) >> ITUR_BT_601_SHIFT);
                row2[4]      = saturate_cast<uchar>((y11 + guv) >> ITUR_BT_601_SHIFT);
                row2[3+bIdx] = saturate_cast<uchar>((y11 + buv) >> ITUR_BT_601_SHIFT);
            }
        }
    }
};

template<int bIdx>
struct YUV420p2RGBA8888Invoker : ParallelLoopBody
{
    uchar * dst_data;
    size_t dst_step;
    int width;
    const uchar* my1, *mu, *mv;
    size_t  stride;
    int ustepIdx, vstepIdx;

    YUV420p2RGBA8888Invoker(uchar * _dst_data, size_t _dst_step, int _dst_width, size_t _stride, const uchar* _y1, const uchar* _u, const uchar* _v, int _ustepIdx, int _vstepIdx)
        : dst_data(_dst_data), dst_step(_dst_step), width(_dst_width), my1(_y1), mu(_u), mv(_v), stride(_stride), ustepIdx(_ustepIdx), vstepIdx(_vstepIdx) {}

    void operator()(const Range& range) const
    {
        int rangeBegin = range.start * 2;
        int rangeEnd = range.end * 2;

        int uvsteps[2] = {width/2, static_cast<int>(stride) - width/2};
        int usIdx = ustepIdx, vsIdx = vstepIdx;

        const uchar* y1 = my1 + rangeBegin * stride;
        const uchar* u1 = mu + (range.start / 2) * stride;
        const uchar* v1 = mv + (range.start / 2) * stride;

        if(range.start % 2 == 1)
        {
            u1 += uvsteps[(usIdx++) & 1];
            v1 += uvsteps[(vsIdx++) & 1];
        }

        for (int j = rangeBegin; j < rangeEnd; j += 2, y1 += stride * 2, u1 += uvsteps[(usIdx++) & 1], v1 += uvsteps[(vsIdx++) & 1])
        {
            uchar* row1 = dst_data + dst_step * j;
            uchar* row2 = dst_data + dst_step * (j + 1);
            const uchar* y2 = y1 + stride;

            for (int i = 0; i < width / 2; i += 1, row1 += 8, row2 += 8)
            {
                int u = int(u1[i]) - 128;
                int v = int(v1[i]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(y1[2 * i]) - 16) * ITUR_BT_601_CY;
                row1[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row1[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row1[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);
                row1[3]      = uchar(0xff);

                int y01 = std::max(0, int(y1[2 * i + 1]) - 16) * ITUR_BT_601_CY;
                row1[6-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row1[5]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row1[4+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);
                row1[7]      = uchar(0xff);

                int y10 = std::max(0, int(y2[2 * i]) - 16) * ITUR_BT_601_CY;
                row2[2-bIdx] = saturate_cast<uchar>((y10 + ruv) >> ITUR_BT_601_SHIFT);
                row2[1]      = saturate_cast<uchar>((y10 + guv) >> ITUR_BT_601_SHIFT);
                row2[bIdx]   = saturate_cast<uchar>((y10 + buv) >> ITUR_BT_601_SHIFT);
                row2[3]      = uchar(0xff);

                int y11 = std::max(0, int(y2[2 * i + 1]) - 16) * ITUR_BT_601_CY;
                row2[6-bIdx] = saturate_cast<uchar>((y11 + ruv) >> ITUR_BT_601_SHIFT);
                row2[5]      = saturate_cast<uchar>((y11 + guv) >> ITUR_BT_601_SHIFT);
                row2[4+bIdx] = saturate_cast<uchar>((y11 + buv) >> ITUR_BT_601_SHIFT);
                row2[7]      = uchar(0xff);
            }
        }
    }
};

#define MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION (320*240)

template<int bIdx, int uIdx>
inline void cvtYUV420sp2RGB(uchar * dst_data, size_t dst_step, int dst_width, int dst_height, size_t _stride, const uchar* _y1, const uchar* _uv)
{
    YUV420sp2RGB888Invoker<bIdx, uIdx> converter(dst_data, dst_step, dst_width, _stride, _y1,  _uv);
    if (dst_width * dst_height >= MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION)
        parallel_for_(Range(0, dst_height/2), converter);
    else
        converter(Range(0, dst_height/2));
}

template<int bIdx, int uIdx>
inline void cvtYUV420sp2RGBA(uchar * dst_data, size_t dst_step, int dst_width, int dst_height, size_t _stride, const uchar* _y1, const uchar* _uv)
{
    YUV420sp2RGBA8888Invoker<bIdx, uIdx> converter(dst_data, dst_step, dst_width, _stride, _y1,  _uv);
    if (dst_width * dst_height >= MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION)
        parallel_for_(Range(0, dst_height/2), converter);
    else
        converter(Range(0, dst_height/2));
}

template<int bIdx>
inline void cvtYUV420p2RGB(uchar * dst_data, size_t dst_step, int dst_width, int dst_height, size_t _stride, const uchar* _y1, const uchar* _u, const uchar* _v, int ustepIdx, int vstepIdx)
{
    YUV420p2RGB888Invoker<bIdx> converter(dst_data, dst_step, dst_width, _stride, _y1,  _u, _v, ustepIdx, vstepIdx);
    if (dst_width * dst_height >= MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION)
        parallel_for_(Range(0, dst_height/2), converter);
    else
        converter(Range(0, dst_height/2));
}

template<int bIdx>
inline void cvtYUV420p2RGBA(uchar * dst_data, size_t dst_step, int dst_width, int dst_height, size_t _stride, const uchar* _y1, const uchar* _u, const uchar* _v, int ustepIdx, int vstepIdx)
{
    YUV420p2RGBA8888Invoker<bIdx> converter(dst_data, dst_step, dst_width, _stride, _y1,  _u, _v, ustepIdx, vstepIdx);
    if (dst_width * dst_height >= MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION)
        parallel_for_(Range(0, dst_height/2), converter);
    else
        converter(Range(0, dst_height/2));
}

///////////////////////////////////// RGB -> YUV420p /////////////////////////////////////

template<int uIdx>
inline void swapUV(uchar * &, uchar * &) {}
template<>
inline void swapUV<2>(uchar * & u, uchar * & v) { std::swap(u, v); }

template<int bIdx, int uIdx>
struct RGB888toYUV420pInvoker: public ParallelLoopBody
{
    RGB888toYUV420pInvoker(const uchar * _src_data, size_t _src_step, uchar * _dst_data, size_t _dst_step,
                           int _src_width, int _src_height, int _scn)
        : src_data(_src_data), src_step(_src_step),
          dst_data(_dst_data), dst_step(_dst_step),
          src_width(_src_width), src_height(_src_height),
          scn(_scn) { }

    void operator()(const Range& rowRange) const
    {
        const int w = src_width;
        const int h = src_height;

        const int cn = scn;
        for( int i = rowRange.start; i < rowRange.end; i++ )
        {
            const uchar* row0 = src_data + src_step * (2 * i);
            const uchar* row1 = src_data + src_step * (2 * i + 1);

            uchar* y = dst_data + dst_step * (2*i);
            uchar* u = dst_data + dst_step * (h + i/2) + (i % 2) * (w/2);
            uchar* v = dst_data + dst_step * (h + (i + h/2)/2) + ((i + h/2) % 2) * (w/2);

            swapUV<uIdx>(u, v);

            for( int j = 0, k = 0; j < w * cn; j += 2 * cn, k++ )
            {
                int r00 = row0[2-bIdx + j];      int g00 = row0[1 + j];      int b00 = row0[bIdx + j];
                int r01 = row0[2-bIdx + cn + j]; int g01 = row0[1 + cn + j]; int b01 = row0[bIdx + cn + j];
                int r10 = row1[2-bIdx + j];      int g10 = row1[1 + j];      int b10 = row1[bIdx + j];
                int r11 = row1[2-bIdx + cn + j]; int g11 = row1[1 + cn + j]; int b11 = row1[bIdx + cn + j];

                const int shifted16 = (16 << ITUR_BT_601_SHIFT);
                const int halfShift = (1 << (ITUR_BT_601_SHIFT - 1));
                int y00 = ITUR_BT_601_CRY * r00 + ITUR_BT_601_CGY * g00 + ITUR_BT_601_CBY * b00 + halfShift + shifted16;
                int y01 = ITUR_BT_601_CRY * r01 + ITUR_BT_601_CGY * g01 + ITUR_BT_601_CBY * b01 + halfShift + shifted16;
                int y10 = ITUR_BT_601_CRY * r10 + ITUR_BT_601_CGY * g10 + ITUR_BT_601_CBY * b10 + halfShift + shifted16;
                int y11 = ITUR_BT_601_CRY * r11 + ITUR_BT_601_CGY * g11 + ITUR_BT_601_CBY * b11 + halfShift + shifted16;

                y[2*k + 0]            = saturate_cast<uchar>(y00 >> ITUR_BT_601_SHIFT);
                y[2*k + 1]            = saturate_cast<uchar>(y01 >> ITUR_BT_601_SHIFT);
                y[2*k + dst_step + 0] = saturate_cast<uchar>(y10 >> ITUR_BT_601_SHIFT);
                y[2*k + dst_step + 1] = saturate_cast<uchar>(y11 >> ITUR_BT_601_SHIFT);

                const int shifted128 = (128 << ITUR_BT_601_SHIFT);
                int u00 = ITUR_BT_601_CRU * r00 + ITUR_BT_601_CGU * g00 + ITUR_BT_601_CBU * b00 + halfShift + shifted128;
                int v00 = ITUR_BT_601_CBU * r00 + ITUR_BT_601_CGV * g00 + ITUR_BT_601_CBV * b00 + halfShift + shifted128;

                u[k] = saturate_cast<uchar>(u00 >> ITUR_BT_601_SHIFT);
                v[k] = saturate_cast<uchar>(v00 >> ITUR_BT_601_SHIFT);
            }
        }
    }

    static bool isFit( int src_width, int src_height )
    {
        return (src_width * src_height >= 320*240);
    }

private:
    RGB888toYUV420pInvoker& operator=(const RGB888toYUV420pInvoker&);

    const uchar * src_data;
    size_t src_step;
    uchar * dst_data;
    size_t dst_step;
    int src_width;
    int src_height;
    const int scn;
};

template<int bIdx, int uIdx>
static void cvtRGBtoYUV420p(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int src_width, int src_height, int scn)
{
    RGB888toYUV420pInvoker<bIdx, uIdx> colorConverter(src_data, src_step, dst_data, dst_step, src_width, src_height, scn);
    if( RGB888toYUV420pInvoker<bIdx, uIdx>::isFit(src_width, src_height) )
        parallel_for_(Range(0, src_height/2), colorConverter);
    else
        colorConverter(Range(0, src_height/2));
}

///////////////////////////////////// YUV422 -> RGB /////////////////////////////////////

template<int bIdx, int uIdx, int yIdx>
struct YUV422toRGB888Invoker : ParallelLoopBody
{
    uchar * dst_data;
    size_t dst_step;
    const uchar * src_data;
    size_t src_step;
    int width;

    YUV422toRGB888Invoker(uchar * _dst_data, size_t _dst_step,
                          const uchar * _src_data, size_t _src_step,
                          int _width)
        : dst_data(_dst_data), dst_step(_dst_step), src_data(_src_data), src_step(_src_step), width(_width) {}

    void operator()(const Range& range) const
    {
        int rangeBegin = range.start;
        int rangeEnd = range.end;

        const int uidx = 1 - yIdx + uIdx * 2;
        const int vidx = (2 + uidx) % 4;
        const uchar* yuv_src = src_data + rangeBegin * src_step;

        for (int j = rangeBegin; j < rangeEnd; j++, yuv_src += src_step)
        {
            uchar* row = dst_data + dst_step * j;

            for (int i = 0; i < 2 * width; i += 4, row += 6)
            {
                int u = int(yuv_src[i + uidx]) - 128;
                int v = int(yuv_src[i + vidx]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(yuv_src[i + yIdx]) - 16) * ITUR_BT_601_CY;
                row[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);

                int y01 = std::max(0, int(yuv_src[i + yIdx + 2]) - 16) * ITUR_BT_601_CY;
                row[5-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row[4]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row[3+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);
            }
        }
    }
};

template<int bIdx, int uIdx, int yIdx>
struct YUV422toRGBA8888Invoker : ParallelLoopBody
{
    uchar * dst_data;
    size_t dst_step;
    const uchar * src_data;
    size_t src_step;
    int width;

    YUV422toRGBA8888Invoker(uchar * _dst_data, size_t _dst_step,
                            const uchar * _src_data, size_t _src_step,
                            int _width)
        : dst_data(_dst_data), dst_step(_dst_step), src_data(_src_data), src_step(_src_step), width(_width) {}

    void operator()(const Range& range) const
    {
        int rangeBegin = range.start;
        int rangeEnd = range.end;

        const int uidx = 1 - yIdx + uIdx * 2;
        const int vidx = (2 + uidx) % 4;
        const uchar* yuv_src = src_data + rangeBegin * src_step;

        for (int j = rangeBegin; j < rangeEnd; j++, yuv_src += src_step)
        {
            uchar* row = dst_data + dst_step * j;

            for (int i = 0; i < 2 * width; i += 4, row += 8)
            {
                int u = int(yuv_src[i + uidx]) - 128;
                int v = int(yuv_src[i + vidx]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(yuv_src[i + yIdx]) - 16) * ITUR_BT_601_CY;
                row[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);
                row[3]      = uchar(0xff);

                int y01 = std::max(0, int(yuv_src[i + yIdx + 2]) - 16) * ITUR_BT_601_CY;
                row[6-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row[5]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row[4+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);
                row[7]      = uchar(0xff);
            }
        }
    }
};

#define MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION (320*240)

template<int bIdx, int uIdx, int yIdx>
inline void cvtYUV422toRGB(uchar * dst_data, size_t dst_step, const uchar * src_data, size_t src_step,
                           int width, int height)
{
    YUV422toRGB888Invoker<bIdx, uIdx, yIdx> converter(dst_data, dst_step, src_data, src_step, width);
    if (width * height >= MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION)
        parallel_for_(Range(0, height), converter);
    else
        converter(Range(0, height));
}

template<int bIdx, int uIdx, int yIdx>
inline void cvtYUV422toRGBA(uchar * dst_data, size_t dst_step, const uchar * src_data, size_t src_step,
                           int width, int height)
{
    YUV422toRGBA8888Invoker<bIdx, uIdx, yIdx> converter(dst_data, dst_step, src_data, src_step, width);
    if (width * height >= MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION)
        parallel_for_(Range(0, height), converter);
    else
        converter(Range(0, height));
}

/////////////////////////// RGBA <-> mRGBA (alpha premultiplied) //////////////

template<typename _Tp>
struct RGBA2mRGBA
{
    typedef _Tp channel_type;

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        _Tp max_val  = ColorChannel<_Tp>::max();
        _Tp half_val = ColorChannel<_Tp>::half();
        for( int i = 0; i < n; i++ )
        {
            _Tp v0 = *src++;
            _Tp v1 = *src++;
            _Tp v2 = *src++;
            _Tp v3 = *src++;

            *dst++ = (v0 * v3 + half_val) / max_val;
            *dst++ = (v1 * v3 + half_val) / max_val;
            *dst++ = (v2 * v3 + half_val) / max_val;
            *dst++ = v3;
        }
    }
};


template<typename _Tp>
struct mRGBA2RGBA
{
    typedef _Tp channel_type;

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        _Tp max_val = ColorChannel<_Tp>::max();
        for( int i = 0; i < n; i++ )
        {
            _Tp v0 = *src++;
            _Tp v1 = *src++;
            _Tp v2 = *src++;
            _Tp v3 = *src++;
            _Tp v3_half = v3 / 2;

            *dst++ = (v3==0)? 0 : (v0 * max_val + v3_half) / v3;
            *dst++ = (v3==0)? 0 : (v1 * max_val + v3_half) / v3;
            *dst++ = (v3==0)? 0 : (v2 * max_val + v3_half) / v3;
            *dst++ = v3;
        }
    }
};

}

//
// HAL functions
//

namespace cv {
namespace hal {

// 8u, 16u, 32f
void cvtBGRtoBGR(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, int dcn, bool swapBlue)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtBGRtoBGR, cv_hal_cvtBGRtoBGR, src_data, src_step, dst_data, dst_step, width, height, depth, scn, dcn, swapBlue);

    int blueIdx = swapBlue ? 2 : 0;
    if( depth == CV_8U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2RGB<uchar>(scn, dcn, blueIdx));
    else if( depth == CV_16U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2RGB<ushort>(scn, dcn, blueIdx));
    else
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2RGB<float>(scn, dcn, blueIdx));
}

// only 8u
void cvtBGRtoBGR5x5(const uchar * src_data, size_t src_step,
                    uchar * dst_data, size_t dst_step,
                    int width, int height,
                    int scn, bool swapBlue, int greenBits)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtBGRtoBGR5x5, cv_hal_cvtBGRtoBGR5x5, src_data, src_step, dst_data, dst_step, width, height, scn, swapBlue, greenBits);

    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2RGB5x5(scn, swapBlue ? 2 : 0, greenBits));
}

// only 8u
void cvtBGR5x5toBGR(const uchar * src_data, size_t src_step,
                    uchar * dst_data, size_t dst_step,
                    int width, int height,
                    int dcn, bool swapBlue, int greenBits)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtBGR5x5toBGR, cv_hal_cvtBGR5x5toBGR, src_data, src_step, dst_data, dst_step, width, height, dcn, swapBlue, greenBits);

    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB5x52RGB(dcn, swapBlue ? 2 : 0, greenBits));
}

// 8u, 16u, 32f
void cvtBGRtoGray(const uchar * src_data, size_t src_step,
                  uchar * dst_data, size_t dst_step,
                  int width, int height,
                  int depth, int scn, bool swapBlue)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtBGRtoGray, cv_hal_cvtBGRtoGray, src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue);

    int blueIdx = swapBlue ? 2 : 0;
    if( depth == CV_8U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Gray<uchar>(scn, blueIdx, 0));
    else if( depth == CV_16U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Gray<ushort>(scn, blueIdx, 0));
    else
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Gray<float>(scn, blueIdx, 0));
}

// 8u, 16u, 32f
void cvtGraytoBGR(const uchar * src_data, size_t src_step,
                  uchar * dst_data, size_t dst_step,
                  int width, int height,
                  int depth, int dcn)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtGraytoBGR, cv_hal_cvtGraytoBGR, src_data, src_step, dst_data, dst_step, width, height, depth, dcn);

    if( depth == CV_8U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Gray2RGB<uchar>(dcn));
    else if( depth == CV_16U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Gray2RGB<ushort>(dcn));
    else
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Gray2RGB<float>(dcn));
}

// only 8u
void cvtBGR5x5toGray(const uchar * src_data, size_t src_step,
                     uchar * dst_data, size_t dst_step,
                     int width, int height,
                     int greenBits)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtBGR5x5toGray, cv_hal_cvtBGR5x5toGray, src_data, src_step, dst_data, dst_step, width, height, greenBits);
    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB5x52Gray(greenBits));
}

// only 8u
void cvtGraytoBGR5x5(const uchar * src_data, size_t src_step,
                     uchar * dst_data, size_t dst_step,
                     int width, int height,
                     int greenBits)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtGraytoBGR5x5, cv_hal_cvtGraytoBGR5x5, src_data, src_step, dst_data, dst_step, width, height, greenBits);
    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Gray2RGB5x5(greenBits));
}

// 8u, 16u, 32f
void cvtBGRtoYUV(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, bool swapBlue, bool isCbCr)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtBGRtoYUV, cv_hal_cvtBGRtoYUV, src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue, isCbCr);

    int blueIdx = swapBlue ? 2 : 0;
    if( depth == CV_8U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2YCrCb_i<uchar>(scn, blueIdx, isCbCr));
    else if( depth == CV_16U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2YCrCb_i<ushort>(scn, blueIdx, isCbCr));
    else
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2YCrCb_f<float>(scn, blueIdx, isCbCr));
}

void cvtYUVtoBGR(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int dcn, bool swapBlue, bool isCbCr)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtYUVtoBGR, cv_hal_cvtYUVtoBGR, src_data, src_step, dst_data, dst_step, width, height, depth, dcn, swapBlue, isCbCr);

    int blueIdx = swapBlue ? 2 : 0;
    if( depth == CV_8U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, YCrCb2RGB_i<uchar>(dcn, blueIdx, isCbCr));
    else if( depth == CV_16U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, YCrCb2RGB_i<ushort>(dcn, blueIdx, isCbCr));
    else
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, YCrCb2RGB_f<float>(dcn, blueIdx, isCbCr));
}

void cvtBGRtoXYZ(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, bool swapBlue)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtBGRtoXYZ, cv_hal_cvtBGRtoXYZ, src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue);

    int blueIdx = swapBlue ? 2 : 0;
    if( depth == CV_8U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2XYZ_i<uchar>(scn, blueIdx, 0));
    else if( depth == CV_16U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2XYZ_i<ushort>(scn, blueIdx, 0));
    else
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2XYZ_f<float>(scn, blueIdx, 0));
}

void cvtXYZtoBGR(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int dcn, bool swapBlue)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtXYZtoBGR, cv_hal_cvtXYZtoBGR, src_data, src_step, dst_data, dst_step, width, height, depth, dcn, swapBlue);

    int blueIdx = swapBlue ? 2 : 0;
    if( depth == CV_8U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, XYZ2RGB_i<uchar>(dcn, blueIdx, 0));
    else if( depth == CV_16U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, XYZ2RGB_i<ushort>(dcn, blueIdx, 0));
    else
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, XYZ2RGB_f<float>(dcn, blueIdx, 0));
}

// 8u, 32f
void cvtBGRtoHSV(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, bool swapBlue, bool isFullRange, bool isHSV)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtBGRtoHSV, cv_hal_cvtBGRtoHSV, src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue, isFullRange, isHSV);

    int hrange = depth == CV_32F ? 360 : isFullRange ? 256 : 180;
    int blueIdx = swapBlue ? 2 : 0;
    if(isHSV)
    {
        if(depth == CV_8U)
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2HSV_b(scn, blueIdx, hrange));
        else
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2HSV_f(scn, blueIdx, static_cast<float>(hrange)));
    }
    else
    {
        if( depth == CV_8U )
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2HLS_b(scn, blueIdx, hrange));
        else
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2HLS_f(scn, blueIdx, static_cast<float>(hrange)));
    }
}

// 8u, 32f
void cvtHSVtoBGR(const uchar * src_data, size_t src_step,
                        uchar * dst_data, size_t dst_step,
                        int width, int height,
                        int depth, int dcn, bool swapBlue, bool isFullRange, bool isHSV)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtHSVtoBGR, cv_hal_cvtHSVtoBGR, src_data, src_step, dst_data, dst_step, width, height, depth, dcn, swapBlue, isFullRange, isHSV);

    int hrange = depth == CV_32F ? 360 : isFullRange ? 255 : 180;
    int blueIdx = swapBlue ? 2 : 0;
    if(isHSV)
    {
        if( depth == CV_8U )
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, HSV2RGB_b(dcn, blueIdx, hrange));
        else
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, HSV2RGB_f(dcn, blueIdx, static_cast<float>(hrange)));
    }
    else
    {
        if( depth == CV_8U )
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, HLS2RGB_b(dcn, blueIdx, hrange));
        else
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, HLS2RGB_f(dcn, blueIdx, static_cast<float>(hrange)));
    }
}

// 8u, 32f
void cvtBGRtoLab(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, bool swapBlue, bool isLab, bool srgb)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtBGRtoLab, cv_hal_cvtBGRtoLab, src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue, isLab, srgb);

    int blueIdx = swapBlue ? 2 : 0;
    if(isLab)
    {
        if( depth == CV_8U )
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Lab_b(scn, blueIdx, 0, 0, srgb));
        else
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Lab_f(scn, blueIdx, 0, 0, srgb));
    }
    else
    {
        if( depth == CV_8U )
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Luv_b(scn, blueIdx, 0, 0, srgb));
        else
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Luv_f(scn, blueIdx, 0, 0, srgb));
    }
}

// 8u, 32f
void cvtLabtoBGR(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int dcn, bool swapBlue, bool isLab, bool srgb)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtLabtoBGR, cv_hal_cvtLabtoBGR, src_data, src_step, dst_data, dst_step, width, height, depth, dcn, swapBlue, isLab, srgb);

    int blueIdx = swapBlue ? 2 : 0;
    if(isLab)
    {
        if( depth == CV_8U )
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Lab2RGB_b(dcn, blueIdx, 0, 0, srgb));
        else
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Lab2RGB_f(dcn, blueIdx, 0, 0, srgb));
    }
    else
    {
        if( depth == CV_8U )
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Luv2RGB_b(dcn, blueIdx, 0, 0, srgb));
        else
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Luv2RGB_f(dcn, blueIdx, 0, 0, srgb));
    }
}

void cvtTwoPlaneYUVtoBGR(const uchar * src_data, size_t src_step,
                                uchar * dst_data, size_t dst_step,
                                int dst_width, int dst_height,
                                int dcn, bool swapBlue, int uIdx)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtTwoPlaneYUVtoBGR, cv_hal_cvtTwoPlaneYUVtoBGR, src_data, src_step, dst_data, dst_step, dst_width, dst_height, dcn, swapBlue, uIdx);
    int blueIdx = swapBlue ? 2 : 0;
    const uchar* uv = src_data + src_step * static_cast<size_t>(dst_height);
    switch(dcn*100 + blueIdx * 10 + uIdx)
    {
    case 300: cvtYUV420sp2RGB<0, 0> (dst_data, dst_step, dst_width, dst_height, src_step, src_data, uv); break;
    case 301: cvtYUV420sp2RGB<0, 1> (dst_data, dst_step, dst_width, dst_height, src_step, src_data, uv); break;
    case 320: cvtYUV420sp2RGB<2, 0> (dst_data, dst_step, dst_width, dst_height, src_step, src_data, uv); break;
    case 321: cvtYUV420sp2RGB<2, 1> (dst_data, dst_step, dst_width, dst_height, src_step, src_data, uv); break;
    case 400: cvtYUV420sp2RGBA<0, 0>(dst_data, dst_step, dst_width, dst_height, src_step, src_data, uv); break;
    case 401: cvtYUV420sp2RGBA<0, 1>(dst_data, dst_step, dst_width, dst_height, src_step, src_data, uv); break;
    case 420: cvtYUV420sp2RGBA<2, 0>(dst_data, dst_step, dst_width, dst_height, src_step, src_data, uv); break;
    case 421: cvtYUV420sp2RGBA<2, 1>(dst_data, dst_step, dst_width, dst_height, src_step, src_data, uv); break;
    default: CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" ); break;
    };
}

void cvtThreePlaneYUVtoBGR(const uchar * src_data, size_t src_step,
                                  uchar * dst_data, size_t dst_step,
                                  int dst_width, int dst_height,
                                  int dcn, bool swapBlue, int uIdx)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtThreePlaneYUVtoBGR, cv_hal_cvtThreePlaneYUVtoBGR, src_data, src_step, dst_data, dst_step, dst_width, dst_height, dcn, swapBlue, uIdx);
    const uchar* u = src_data + src_step * static_cast<size_t>(dst_height);
    const uchar* v = src_data + src_step * static_cast<size_t>(dst_height + dst_height/4) + (dst_width/2) * ((dst_height % 4)/2);

    int ustepIdx = 0;
    int vstepIdx = dst_height % 4 == 2 ? 1 : 0;

    if(uIdx == 1) { std::swap(u ,v), std::swap(ustepIdx, vstepIdx); }
    int blueIdx = swapBlue ? 2 : 0;

    switch(dcn*10 + blueIdx)
    {
    case 30: cvtYUV420p2RGB<0>(dst_data, dst_step, dst_width, dst_height, src_step, src_data, u, v, ustepIdx, vstepIdx); break;
    case 32: cvtYUV420p2RGB<2>(dst_data, dst_step, dst_width, dst_height, src_step, src_data, u, v, ustepIdx, vstepIdx); break;
    case 40: cvtYUV420p2RGBA<0>(dst_data, dst_step, dst_width, dst_height, src_step, src_data, u, v, ustepIdx, vstepIdx); break;
    case 42: cvtYUV420p2RGBA<2>(dst_data, dst_step, dst_width, dst_height, src_step, src_data, u, v, ustepIdx, vstepIdx); break;
    default: CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" ); break;
    };
}

void cvtBGRtoThreePlaneYUV(const uchar * src_data, size_t src_step,
                           uchar * dst_data, size_t dst_step,
                           int width, int height,
                           int scn, bool swapBlue, int uIdx)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtBGRtoThreePlaneYUV, cv_hal_cvtBGRtoThreePlaneYUV, src_data, src_step, dst_data, dst_step, width, height, scn, swapBlue, uIdx);
    int blueIdx = swapBlue ? 2 : 0;
    switch(blueIdx + uIdx*10)
    {
    case 10: cvtRGBtoYUV420p<0, 1>(src_data, src_step, dst_data, dst_step, width, height, scn); break;
    case 12: cvtRGBtoYUV420p<2, 1>(src_data, src_step, dst_data, dst_step, width, height, scn); break;
    case 20: cvtRGBtoYUV420p<0, 2>(src_data, src_step, dst_data, dst_step, width, height, scn); break;
    case 22: cvtRGBtoYUV420p<2, 2>(src_data, src_step, dst_data, dst_step, width, height, scn); break;
    default: CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" ); break;
    };
}

void cvtOnePlaneYUVtoBGR(const uchar * src_data, size_t src_step,
                         uchar * dst_data, size_t dst_step,
                         int width, int height,
                         int dcn, bool swapBlue, int uIdx, int ycn)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtOnePlaneYUVtoBGR, cv_hal_cvtOnePlaneYUVtoBGR, src_data, src_step, dst_data, dst_step, width, height, dcn, swapBlue, uIdx, ycn);
    int blueIdx = swapBlue ? 2 : 0;
    switch(dcn*1000 + blueIdx*100 + uIdx*10 + ycn)
    {
    case 3000: cvtYUV422toRGB<0,0,0>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 3001: cvtYUV422toRGB<0,0,1>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 3010: cvtYUV422toRGB<0,1,0>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 3200: cvtYUV422toRGB<2,0,0>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 3201: cvtYUV422toRGB<2,0,1>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 3210: cvtYUV422toRGB<2,1,0>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 4000: cvtYUV422toRGBA<0,0,0>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 4001: cvtYUV422toRGBA<0,0,1>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 4010: cvtYUV422toRGBA<0,1,0>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 4200: cvtYUV422toRGBA<2,0,0>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 4201: cvtYUV422toRGBA<2,0,1>(dst_data, dst_step, src_data, src_step, width, height); break;
    case 4210: cvtYUV422toRGBA<2,1,0>(dst_data, dst_step, src_data, src_step, width, height); break;
    default: CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" ); break;
    };
}

void cvtRGBAtoMultipliedRGBA(const uchar * src_data, size_t src_step,
                             uchar * dst_data, size_t dst_step,
                             int width, int height)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtRGBAtoMultipliedRGBA, cv_hal_cvtRGBAtoMultipliedRGBA, src_data, src_step, dst_data, dst_step, width, height);

    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGBA2mRGBA<uchar>());
}

void cvtMultipliedRGBAtoRGBA(const uchar * src_data, size_t src_step,
                             uchar * dst_data, size_t dst_step,
                             int width, int height)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(cvtMultipliedRGBAtoRGBA, cv_hal_cvtMultipliedRGBAtoRGBA, src_data, src_step, dst_data, dst_step, width, height);
    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, mRGBA2RGBA<uchar>());
}

}} // cv::hal::

//
// Helper functions
//

inline bool isHSV(int code)
{
    switch(code)
    {
    case CV_HSV2BGR: case CV_HSV2RGB: case CV_HSV2BGR_FULL: case CV_HSV2RGB_FULL:
    case CV_BGR2HSV: case CV_RGB2HSV: case CV_BGR2HSV_FULL: case CV_RGB2HSV_FULL:
        return true;
    default:
        return false;
    }
}

inline bool isLab(int code)
{
    switch (code)
    {
    case CV_Lab2BGR: case CV_Lab2RGB: case CV_Lab2LBGR: case CV_Lab2LRGB:
    case CV_BGR2Lab: case CV_RGB2Lab: case CV_LBGR2Lab: case CV_LRGB2Lab:
        return true;
    default:
        return false;
    }
}

inline bool issRGB(int code)
{
    switch (code)
    {
    case CV_BGR2Lab: case CV_RGB2Lab: case CV_BGR2Luv: case CV_RGB2Luv:
    case CV_Lab2BGR: case CV_Lab2RGB: case CV_Luv2BGR: case CV_Luv2RGB:
        return true;
    default:
        return false;
    }
}

inline bool swapBlue(int code)
{
    switch (code)
    {
    case CV_BGR2BGRA: case CV_BGRA2BGR:
    case CV_BGR2BGR565: case CV_BGR2BGR555: case CV_BGRA2BGR565: case CV_BGRA2BGR555:
    case CV_BGR5652BGR: case CV_BGR5552BGR: case CV_BGR5652BGRA: case CV_BGR5552BGRA:
    case CV_BGR2GRAY: case CV_BGRA2GRAY:
    case CV_BGR2YCrCb: case CV_BGR2YUV:
    case CV_YCrCb2BGR: case CV_YUV2BGR:
    case CV_BGR2XYZ: case CV_XYZ2BGR:
    case CV_BGR2HSV: case CV_BGR2HLS: case CV_BGR2HSV_FULL: case CV_BGR2HLS_FULL:
    case CV_YUV2BGR_YV12: case CV_YUV2BGRA_YV12: case CV_YUV2BGR_IYUV: case CV_YUV2BGRA_IYUV:
    case CV_YUV2BGR_NV21: case CV_YUV2BGRA_NV21: case CV_YUV2BGR_NV12: case CV_YUV2BGRA_NV12:
    case CV_Lab2BGR: case CV_Luv2BGR: case CV_Lab2LBGR: case CV_Luv2LBGR:
    case CV_BGR2Lab: case CV_BGR2Luv: case CV_LBGR2Lab: case CV_LBGR2Luv:
    case CV_HSV2BGR: case CV_HLS2BGR: case CV_HSV2BGR_FULL: case CV_HLS2BGR_FULL:
    case CV_YUV2BGR_UYVY: case CV_YUV2BGRA_UYVY: case CV_YUV2BGR_YUY2:
    case CV_YUV2BGRA_YUY2:  case CV_YUV2BGR_YVYU: case CV_YUV2BGRA_YVYU:
    case CV_BGR2YUV_IYUV: case CV_BGRA2YUV_IYUV: case CV_BGR2YUV_YV12: case CV_BGRA2YUV_YV12:
        return false;
    default:
        return true;
    }
}

inline bool isFullRange(int code)
{
    switch (code)
    {
    case CV_BGR2HSV_FULL: case CV_RGB2HSV_FULL: case CV_BGR2HLS_FULL: case CV_RGB2HLS_FULL:
    case CV_HSV2BGR_FULL: case CV_HSV2RGB_FULL: case CV_HLS2BGR_FULL: case CV_HLS2RGB_FULL:
        return true;
    default:
        return false;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
//                                   The main function                                  //
//////////////////////////////////////////////////////////////////////////////////////////

void cv::cvtColor( InputArray _src, OutputArray _dst, int code, int dcn )
{
    CV_INSTRUMENT_REGION()

    int stype = _src.type();
    int scn = CV_MAT_CN(stype), depth = CV_MAT_DEPTH(stype), uidx, gbits, ycn;

    Mat src, dst;
    if (_src.getObj() == _dst.getObj()) // inplace processing (#6653)
        _src.copyTo(src);
    else
        src = _src.getMat();
    Size sz = src.size();
    CV_Assert( depth == CV_8U || depth == CV_16U || depth == CV_32F );

    switch( code )
    {
        case CV_BGR2BGRA: case CV_RGB2BGRA: case CV_BGRA2BGR:
        case CV_RGBA2BGR: case CV_RGB2BGR: case CV_BGRA2RGBA:
            CV_Assert( scn == 3 || scn == 4 );
            dcn = code == CV_BGR2BGRA || code == CV_RGB2BGRA || code == CV_BGRA2RGBA ? 4 : 3;
            _dst.create( sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtBGRtoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                             depth, scn, dcn, swapBlue(code));
            break;

        case CV_BGR2BGR565: case CV_BGR2BGR555: case CV_RGB2BGR565: case CV_RGB2BGR555:
        case CV_BGRA2BGR565: case CV_BGRA2BGR555: case CV_RGBA2BGR565: case CV_RGBA2BGR555:
            CV_Assert( (scn == 3 || scn == 4) && depth == CV_8U );
            gbits = code == CV_BGR2BGR565 || code == CV_RGB2BGR565 ||
                    code == CV_BGRA2BGR565 || code == CV_RGBA2BGR565 ? 6 : 5;
            _dst.create(sz, CV_8UC2);
            dst = _dst.getMat();
            hal::cvtBGRtoBGR5x5(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                                scn, swapBlue(code), gbits);
            break;

        case CV_BGR5652BGR: case CV_BGR5552BGR: case CV_BGR5652RGB: case CV_BGR5552RGB:
        case CV_BGR5652BGRA: case CV_BGR5552BGRA: case CV_BGR5652RGBA: case CV_BGR5552RGBA:
            if(dcn <= 0) dcn = (code==CV_BGR5652BGRA || code==CV_BGR5552BGRA || code==CV_BGR5652RGBA || code==CV_BGR5552RGBA) ? 4 : 3;
            CV_Assert( (dcn == 3 || dcn == 4) && scn == 2 && depth == CV_8U );
            gbits = code == CV_BGR5652BGR || code == CV_BGR5652RGB ||
                    code == CV_BGR5652BGRA || code == CV_BGR5652RGBA ? 6 : 5;
            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtBGR5x5toBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                                dcn, swapBlue(code), gbits);
            break;

        case CV_BGR2GRAY: case CV_BGRA2GRAY: case CV_RGB2GRAY: case CV_RGBA2GRAY:
            CV_Assert( scn == 3 || scn == 4 );
            _dst.create(sz, CV_MAKETYPE(depth, 1));
            dst = _dst.getMat();
            hal::cvtBGRtoGray(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                              depth, scn, swapBlue(code));
            break;

        case CV_BGR5652GRAY: case CV_BGR5552GRAY:
            CV_Assert( scn == 2 && depth == CV_8U );
            gbits = code == CV_BGR5652GRAY ? 6 : 5;
            _dst.create(sz, CV_8UC1);
            dst = _dst.getMat();
            hal::cvtBGR5x5toGray(src.data, src.step, dst.data, dst.step, src.cols, src.rows, gbits);
            break;

        case CV_GRAY2BGR: case CV_GRAY2BGRA:
            if( dcn <= 0 ) dcn = (code==CV_GRAY2BGRA) ? 4 : 3;
            CV_Assert( scn == 1 && (dcn == 3 || dcn == 4));
            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtGraytoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows, depth, dcn);
            break;

        case CV_GRAY2BGR565: case CV_GRAY2BGR555:
            CV_Assert( scn == 1 && depth == CV_8U );
            gbits = code == CV_GRAY2BGR565 ? 6 : 5;
            _dst.create(sz, CV_8UC2);
            dst = _dst.getMat();
            hal::cvtGraytoBGR5x5(src.data, src.step, dst.data, dst.step, src.cols, src.rows, gbits);
            break;

        case CV_BGR2YCrCb: case CV_RGB2YCrCb:
        case CV_BGR2YUV: case CV_RGB2YUV:
            CV_Assert( scn == 3 || scn == 4 );
            _dst.create(sz, CV_MAKETYPE(depth, 3));
            dst = _dst.getMat();
            hal::cvtBGRtoYUV(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                             depth, scn, swapBlue(code), code == CV_BGR2YCrCb || code == CV_RGB2YCrCb);
            break;

        case CV_YCrCb2BGR: case CV_YCrCb2RGB:
        case CV_YUV2BGR: case CV_YUV2RGB:
            if( dcn <= 0 ) dcn = 3;
            CV_Assert( scn == 3 && (dcn == 3 || dcn == 4) );
            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtYUVtoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                             depth, dcn, swapBlue(code), code == CV_YCrCb2BGR || code == CV_YCrCb2RGB);
            break;

        case CV_BGR2XYZ: case CV_RGB2XYZ:
            CV_Assert( scn == 3 || scn == 4 );
            _dst.create(sz, CV_MAKETYPE(depth, 3));
            dst = _dst.getMat();
            hal::cvtBGRtoXYZ(src.data, src.step, dst.data, dst.step, src.cols, src.rows, depth, scn, swapBlue(code));
            break;

        case CV_XYZ2BGR: case CV_XYZ2RGB:
            if( dcn <= 0 ) dcn = 3;
            CV_Assert( scn == 3 && (dcn == 3 || dcn == 4) );
            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtXYZtoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows, depth, dcn, swapBlue(code));
            break;

        case CV_BGR2HSV: case CV_RGB2HSV: case CV_BGR2HSV_FULL: case CV_RGB2HSV_FULL:
        case CV_BGR2HLS: case CV_RGB2HLS: case CV_BGR2HLS_FULL: case CV_RGB2HLS_FULL:
            CV_Assert( (scn == 3 || scn == 4) && (depth == CV_8U || depth == CV_32F) );
            _dst.create(sz, CV_MAKETYPE(depth, 3));
            dst = _dst.getMat();
            hal::cvtBGRtoHSV(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                             depth, scn, swapBlue(code), isFullRange(code), isHSV(code));
            break;

        case CV_HSV2BGR: case CV_HSV2RGB: case CV_HSV2BGR_FULL: case CV_HSV2RGB_FULL:
        case CV_HLS2BGR: case CV_HLS2RGB: case CV_HLS2BGR_FULL: case CV_HLS2RGB_FULL:
            if( dcn <= 0 ) dcn = 3;
            CV_Assert( scn == 3 && (dcn == 3 || dcn == 4) && (depth == CV_8U || depth == CV_32F) );
            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtHSVtoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                             depth, dcn, swapBlue(code), isFullRange(code), isHSV(code));
            break;

        case CV_BGR2Lab: case CV_RGB2Lab: case CV_LBGR2Lab: case CV_LRGB2Lab:
        case CV_BGR2Luv: case CV_RGB2Luv: case CV_LBGR2Luv: case CV_LRGB2Luv:
            CV_Assert( (scn == 3 || scn == 4) && (depth == CV_8U || depth == CV_32F) );
            _dst.create(sz, CV_MAKETYPE(depth, 3));
            dst = _dst.getMat();
            hal::cvtBGRtoLab(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                             depth, scn, swapBlue(code), isLab(code), issRGB(code));
            break;

        case CV_Lab2BGR: case CV_Lab2RGB: case CV_Lab2LBGR: case CV_Lab2LRGB:
        case CV_Luv2BGR: case CV_Luv2RGB: case CV_Luv2LBGR: case CV_Luv2LRGB:
            if( dcn <= 0 ) dcn = 3;
            CV_Assert( scn == 3 && (dcn == 3 || dcn == 4) && (depth == CV_8U || depth == CV_32F) );
            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtLabtoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                             depth, dcn, swapBlue(code), isLab(code), issRGB(code));
            break;

        case CV_BayerBG2GRAY: case CV_BayerGB2GRAY: case CV_BayerRG2GRAY: case CV_BayerGR2GRAY:
        case CV_BayerBG2BGR: case CV_BayerGB2BGR: case CV_BayerRG2BGR: case CV_BayerGR2BGR:
        case CV_BayerBG2BGR_VNG: case CV_BayerGB2BGR_VNG: case CV_BayerRG2BGR_VNG: case CV_BayerGR2BGR_VNG:
        case CV_BayerBG2BGR_EA: case CV_BayerGB2BGR_EA: case CV_BayerRG2BGR_EA: case CV_BayerGR2BGR_EA:
            demosaicing(src, _dst, code, dcn);
            break;

        case CV_YUV2BGR_NV21:  case CV_YUV2RGB_NV21:  case CV_YUV2BGR_NV12:  case CV_YUV2RGB_NV12:
        case CV_YUV2BGRA_NV21: case CV_YUV2RGBA_NV21: case CV_YUV2BGRA_NV12: case CV_YUV2RGBA_NV12:
            // http://www.fourcc.org/yuv.php#NV21 == yuv420sp -> a plane of 8 bit Y samples followed by an interleaved V/U plane containing 8 bit 2x2 subsampled chroma samples
            // http://www.fourcc.org/yuv.php#NV12 -> a plane of 8 bit Y samples followed by an interleaved U/V plane containing 8 bit 2x2 subsampled colour difference samples
            if (dcn <= 0) dcn = (code==CV_YUV420sp2BGRA || code==CV_YUV420sp2RGBA || code==CV_YUV2BGRA_NV12 || code==CV_YUV2RGBA_NV12) ? 4 : 3;
            uidx = (code==CV_YUV2BGR_NV21 || code==CV_YUV2BGRA_NV21 || code==CV_YUV2RGB_NV21 || code==CV_YUV2RGBA_NV21) ? 1 : 0;
            CV_Assert( dcn == 3 || dcn == 4 );
            CV_Assert( sz.width % 2 == 0 && sz.height % 3 == 0 && depth == CV_8U );
            _dst.create(Size(sz.width, sz.height * 2 / 3), CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtTwoPlaneYUVtoBGR(src.data, src.step, dst.data, dst.step, dst.cols, dst.rows,
                                     dcn, swapBlue(code), uidx);
            break;
        case CV_YUV2BGR_YV12: case CV_YUV2RGB_YV12: case CV_YUV2BGRA_YV12: case CV_YUV2RGBA_YV12:
        case CV_YUV2BGR_IYUV: case CV_YUV2RGB_IYUV: case CV_YUV2BGRA_IYUV: case CV_YUV2RGBA_IYUV:
            //http://www.fourcc.org/yuv.php#YV12 == yuv420p -> It comprises an NxM Y plane followed by (N/2)x(M/2) V and U planes.
            //http://www.fourcc.org/yuv.php#IYUV == I420 -> It comprises an NxN Y plane followed by (N/2)x(N/2) U and V planes
            if (dcn <= 0) dcn = (code==CV_YUV2BGRA_YV12 || code==CV_YUV2RGBA_YV12 || code==CV_YUV2RGBA_IYUV || code==CV_YUV2BGRA_IYUV) ? 4 : 3;
            uidx  = (code==CV_YUV2BGR_YV12 || code==CV_YUV2RGB_YV12 || code==CV_YUV2BGRA_YV12 || code==CV_YUV2RGBA_YV12) ? 1 : 0;
            CV_Assert( dcn == 3 || dcn == 4 );
            CV_Assert( sz.width % 2 == 0 && sz.height % 3 == 0 && depth == CV_8U );
            _dst.create(Size(sz.width, sz.height * 2 / 3), CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtThreePlaneYUVtoBGR(src.data, src.step, dst.data, dst.step, dst.cols, dst.rows,
                                       dcn, swapBlue(code), uidx);
            break;

        case CV_YUV2GRAY_420:
            {
                if (dcn <= 0) dcn = 1;

                CV_Assert( dcn == 1 );
                CV_Assert( sz.width % 2 == 0 && sz.height % 3 == 0 && depth == CV_8U );

                Size dstSz(sz.width, sz.height * 2 / 3);
                _dst.create(dstSz, CV_MAKETYPE(depth, dcn));
                dst = _dst.getMat();

                src(Range(0, dstSz.height), Range::all()).copyTo(dst);
            }
            break;
        case CV_RGB2YUV_YV12: case CV_BGR2YUV_YV12: case CV_RGBA2YUV_YV12: case CV_BGRA2YUV_YV12:
        case CV_RGB2YUV_IYUV: case CV_BGR2YUV_IYUV: case CV_RGBA2YUV_IYUV: case CV_BGRA2YUV_IYUV:
            if (dcn <= 0) dcn = 1;
            uidx = (code == CV_BGR2YUV_IYUV || code == CV_BGRA2YUV_IYUV || code == CV_RGB2YUV_IYUV || code == CV_RGBA2YUV_IYUV) ? 1 : 2;
            CV_Assert( (scn == 3 || scn == 4) && depth == CV_8U );
            CV_Assert( dcn == 1 );
            CV_Assert( sz.width % 2 == 0 && sz.height % 2 == 0 );
            _dst.create(Size(sz.width, sz.height / 2 * 3), CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtBGRtoThreePlaneYUV(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                                       scn, swapBlue(code), uidx);
            break;
        case CV_YUV2RGB_UYVY: case CV_YUV2BGR_UYVY: case CV_YUV2RGBA_UYVY: case CV_YUV2BGRA_UYVY:
        case CV_YUV2RGB_YUY2: case CV_YUV2BGR_YUY2: case CV_YUV2RGB_YVYU: case CV_YUV2BGR_YVYU:
        case CV_YUV2RGBA_YUY2: case CV_YUV2BGRA_YUY2: case CV_YUV2RGBA_YVYU: case CV_YUV2BGRA_YVYU:
            //http://www.fourcc.org/yuv.php#UYVY
            //http://www.fourcc.org/yuv.php#YUY2
            //http://www.fourcc.org/yuv.php#YVYU
            if (dcn <= 0) dcn = (code==CV_YUV2RGBA_UYVY || code==CV_YUV2BGRA_UYVY || code==CV_YUV2RGBA_YUY2 || code==CV_YUV2BGRA_YUY2 || code==CV_YUV2RGBA_YVYU || code==CV_YUV2BGRA_YVYU) ? 4 : 3;
            ycn  = (code==CV_YUV2RGB_UYVY || code==CV_YUV2BGR_UYVY || code==CV_YUV2RGBA_UYVY || code==CV_YUV2BGRA_UYVY) ? 1 : 0;
            uidx = (code==CV_YUV2RGB_YVYU || code==CV_YUV2BGR_YVYU || code==CV_YUV2RGBA_YVYU || code==CV_YUV2BGRA_YVYU) ? 1 : 0;
            CV_Assert( dcn == 3 || dcn == 4 );
            CV_Assert( scn == 2 && depth == CV_8U );
            _dst.create(sz, CV_8UC(dcn));
            dst = _dst.getMat();
            hal::cvtOnePlaneYUVtoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                                     dcn, swapBlue(code), uidx, ycn);
            break;
        case CV_YUV2GRAY_UYVY: case CV_YUV2GRAY_YUY2:
            {
                if (dcn <= 0) dcn = 1;

                CV_Assert( dcn == 1 );
                CV_Assert( scn == 2 && depth == CV_8U );

                src.release(); // T-API datarace fixup
                extractChannel(_src, _dst, code == CV_YUV2GRAY_UYVY ? 1 : 0);
            }
            break;
        case CV_RGBA2mRGBA:
            if (dcn <= 0) dcn = 4;
            CV_Assert( scn == 4 && dcn == 4 && depth == CV_8U );
            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtRGBAtoMultipliedRGBA(src.data, src.step, dst.data, dst.step, src.cols, src.rows);
            break;
        case CV_mRGBA2RGBA:
            if (dcn <= 0) dcn = 4;
            CV_Assert( scn == 4 && dcn == 4 && depth == CV_8U );
            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtMultipliedRGBAtoRGBA(src.data, src.step, dst.data, dst.step, src.cols, src.rows);
            break;
        default:
            CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" );
        }
}

CV_IMPL void
cvCvtColor( const CvArr* srcarr, CvArr* dstarr, int code )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst0 = cv::cvarrToMat(dstarr), dst = dst0;
    CV_Assert( src.depth() == dst.depth() );

    cv::cvtColor(src, dst, code, dst.channels());
    CV_Assert( dst.data == dst0.data );
}


/* End of file. */
