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
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

/*
 * This file includes the code, contributed by Simon Perreault
 * (the function icvMedianBlur_8u_O1)
 *
 * Constant-time median filtering -- http://nomis80.org/ctmf.html
 * Copyright (C) 2006 Simon Perreault
 *
 * Contact:
 *  Laboratoire de vision et systemes numeriques
 *  Pavillon Adrien-Pouliot
 *  Universite Laval
 *  Sainte-Foy, Quebec, Canada
 *  G1K 7P4
 *
 *  perreaul@gel.ulaval.ca
 */

/****************************************************************************************\
                                     Gaussian Blur
\****************************************************************************************/

cv::Mat cv::getGaussianKernel( int n, double sigma, int ktype )
{
    const int SMALL_GAUSSIAN_SIZE = 7;
    static const float small_gaussian_tab[][SMALL_GAUSSIAN_SIZE] =
    {
        {1.f},
        {0.25f, 0.5f, 0.25f},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
        {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
    };

    const float* fixed_kernel = n % 2 == 1 && n <= SMALL_GAUSSIAN_SIZE && sigma <= 0 ?
        small_gaussian_tab[n>>1] : 0;

    CV_Assert( ktype == CV_32F || ktype == CV_64F );
    Mat kernel(n, 1, ktype);
    float* cf = kernel.ptr<float>();
    double* cd = kernel.ptr<double>();

    double sigmaX = sigma > 0 ? sigma : ((n-1)*0.5 - 1)*0.3 + 0.8;
    double scale2X = -0.5/(sigmaX*sigmaX);
    double sum = 0;

    int i;
    for( i = 0; i < n; i++ )
    {
        double x = i - (n-1)*0.5;
        double t = fixed_kernel ? (double)fixed_kernel[i] : std::exp(scale2X*x*x);
        if( ktype == CV_32F )
        {
            cf[i] = (float)t;
            sum += cf[i];
        }
        else
        {
            cd[i] = t;
            sum += cd[i];
        }
    }

    sum = 1./sum;
    for( i = 0; i < n; i++ )
    {
        if( ktype == CV_32F )
            cf[i] = (float)(cf[i]*sum);
        else
            cd[i] *= sum;
    }

    return kernel;
}

namespace cv {

static void createGaussianKernels( Mat & kx, Mat & ky, int type, Size ksize,
                                   double sigma1, double sigma2 )
{
    int depth = CV_MAT_DEPTH(type);
    if( sigma2 <= 0 )
        sigma2 = sigma1;

    // automatic detection of kernel size from sigma
    if( ksize.width <= 0 && sigma1 > 0 )
        ksize.width = cvRound(sigma1*(depth == CV_8U ? 3 : 4)*2 + 1)|1;
    if( ksize.height <= 0 && sigma2 > 0 )
        ksize.height = cvRound(sigma2*(depth == CV_8U ? 3 : 4)*2 + 1)|1;

    CV_Assert( ksize.width > 0 && ksize.width % 2 == 1 &&
        ksize.height > 0 && ksize.height % 2 == 1 );

    sigma1 = std::max( sigma1, 0. );
    sigma2 = std::max( sigma2, 0. );

    kx = getGaussianKernel( ksize.width, sigma1, std::max(depth, CV_32F) );
    if( ksize.height == ksize.width && std::abs(sigma1 - sigma2) < DBL_EPSILON )
        ky = kx;
    else
        ky = getGaussianKernel( ksize.height, sigma2, std::max(depth, CV_32F) );
}

}

cv::Ptr<cv::FilterEngine> cv::createGaussianFilter( int type, Size ksize,
                                        double sigma1, double sigma2,
                                        int borderType )
{
    Mat kx, ky;
    createGaussianKernels(kx, ky, type, ksize, sigma1, sigma2);

    return createSeparableLinearFilter( type, type, kx, ky, Point(-1,-1), 0, borderType );
}


void cv::GaussianBlur( InputArray _src, OutputArray _dst, Size ksize,
                   double sigma1, double sigma2,
                   int borderType )
{
    CV_INSTRUMENT_REGION()

    int type = _src.type();
    Size size = _src.size();
    _dst.create( size, type );

    if( borderType != BORDER_CONSTANT && (borderType & BORDER_ISOLATED) != 0 )
    {
        if( size.height == 1 )
            ksize.height = 1;
        if( size.width == 1 )
            ksize.width = 1;
    }

    if( ksize.width == 1 && ksize.height == 1 )
    {
        _src.copyTo(_dst);
        return;
    }

    CV_IPP_RUN(!(ocl::useOpenCL() && _dst.isUMat()), ipp_GaussianBlur( _src,  _dst,  ksize, sigma1,  sigma2, borderType));

    Mat kx, ky;
    createGaussianKernels(kx, ky, type, ksize, sigma1, sigma2);

    sepFilter2D(_src, _dst, CV_MAT_DEPTH(type), kx, ky, Point(-1,-1), 0, borderType );
}

//////////////////////////////////////////////////////////////////////////////////////////

CV_IMPL void
cvSmooth( const void* srcarr, void* dstarr, int smooth_type,
          int param1, int param2, double param3, double param4 )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst0 = cv::cvarrToMat(dstarr), dst = dst0;

    CV_Assert( dst.size() == src.size() &&
        (smooth_type == CV_BLUR_NO_SCALE || dst.type() == src.type()) );

    if( param2 <= 0 )
        param2 = param1;

    if( smooth_type == CV_GAUSSIAN )
        cv::GaussianBlur( src, dst, cv::Size(param1, param2), param3, param4, cv::BORDER_REPLICATE );
    else
    {
        CV_Assert(false);
    }

    if( dst.data != dst0.data )
        CV_Error( CV_StsUnmatchedFormats, "The destination image does not have the proper type" );
}

/* End of file. */
