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

#if !CV_NEON

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

#endif // !CV_NEON

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
    else
    {
        CV_Assert(false);
    }
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
    else
    {
        CV_Assert(false);
    }
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

        case CV_BGR2GRAY: case CV_BGRA2GRAY: case CV_RGB2GRAY: case CV_RGBA2GRAY:
            CV_Assert( scn == 3 || scn == 4 );
            _dst.create(sz, CV_MAKETYPE(depth, 1));
            dst = _dst.getMat();
            hal::cvtBGRtoGray(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                              depth, scn, swapBlue(code));
            break;

        case CV_BayerBG2GRAY: case CV_BayerGB2GRAY: case CV_BayerRG2GRAY: case CV_BayerGR2GRAY:
        case CV_BayerBG2BGR: case CV_BayerGB2BGR: case CV_BayerRG2BGR: case CV_BayerGR2BGR:
        case CV_BayerBG2BGR_VNG: case CV_BayerGB2BGR_VNG: case CV_BayerRG2BGR_VNG: case CV_BayerGR2BGR_VNG:
        case CV_BayerBG2BGR_EA: case CV_BayerGB2BGR_EA: case CV_BayerRG2BGR_EA: case CV_BayerGR2BGR_EA:
            demosaicing(src, _dst, code, dcn);
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
        case CV_YUV2GRAY_UYVY: case CV_YUV2GRAY_YUY2:
            {
                if (dcn <= 0) dcn = 1;

                CV_Assert( dcn == 1 );
                CV_Assert( scn == 2 && depth == CV_8U );

                src.release(); // T-API datarace fixup
                extractChannel(_src, _dst, code == CV_YUV2GRAY_UYVY ? 1 : 0);
            }
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
