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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_OPTIM_HPP
#define OPENCV_OPTIM_HPP

#include "opencv2/core.hpp"

namespace cv
{

//! return codes for cv::solveLP() function
enum SolveLPResult
{
    SOLVELP_UNBOUNDED    = -2, //!< problem is unbounded (target function can achieve arbitrary high values)
    SOLVELP_UNFEASIBLE    = -1, //!< problem is unfeasible (there are no points that satisfy all the constraints imposed)
    SOLVELP_SINGLE    = 0, //!< there is only one maximum for target function
    SOLVELP_MULTI    = 1 //!< there are multiple maxima for target function - the arbitrary one is returned
};

/** @brief Solve given (non-integer) linear programming problem using the Simplex Algorithm (Simplex Method).

What we mean here by "linear programming problem" (or LP problem, for short) can be formulated as:

\f[\mbox{Maximize } c\cdot x\\
 \mbox{Subject to:}\\
 Ax\leq b\\
 x\geq 0\f]

Where \f$c\f$ is fixed `1`-by-`n` row-vector, \f$A\f$ is fixed `m`-by-`n` matrix, \f$b\f$ is fixed `m`-by-`1`
column vector and \f$x\f$ is an arbitrary `n`-by-`1` column vector, which satisfies the constraints.

Simplex algorithm is one of many algorithms that are designed to handle this sort of problems
efficiently. Although it is not optimal in theoretical sense (there exist algorithms that can solve
any problem written as above in polynomial time, while simplex method degenerates to exponential
time for some special cases), it is well-studied, easy to implement and is shown to work well for
real-life purposes.

The particular implementation is taken almost verbatim from **Introduction to Algorithms, third
edition** by T. H. Cormen, C. E. Leiserson, R. L. Rivest and Clifford Stein. In particular, the
Bland's rule <http://en.wikipedia.org/wiki/Bland%27s_rule> is used to prevent cycling.

@param Func This row-vector corresponds to \f$c\f$ in the LP problem formulation (see above). It should
contain 32- or 64-bit floating point numbers. As a convenience, column-vector may be also submitted,
in the latter case it is understood to correspond to \f$c^T\f$.
@param Constr `m`-by-`n+1` matrix, whose rightmost column corresponds to \f$b\f$ in formulation above
and the remaining to \f$A\f$. It should containt 32- or 64-bit floating point numbers.
@param z The solution will be returned here as a column-vector - it corresponds to \f$c\f$ in the
formulation above. It will contain 64-bit floating point numbers.
@return One of cv::SolveLPResult
 */
CV_EXPORTS_W int solveLP(const Mat& Func, const Mat& Constr, Mat& z);

//! @}

}// cv

#endif
