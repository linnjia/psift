/* 
 *  fish2para_coords_mex
 *
 *  For the given fisheye image model, first the x,y coordinates on the 
 *  fisheye image plane that map to coordinates on the parabolic image
 * 
 * Copyright (C) 2013 Peter Hansen [phansen(at)qatar.cmu.edu]
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR 'AS IS'.
 */

 
#include "mex.h"
#include <math.h>
#include <matrix.h>
#include <stdio.h>

/*
 * Gateway function
 */
void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *u_fish, *v_fish;        /* Output coordinates */ 
    double *mask, *mask_double;     /* Output mask */
    int nr, nc;                     /* Size of input (and output) image */
    double Cx, Cy, m, l;            /* Fisheye image model */
    double *m_para;                 /* Parabolic model scaling factor */
    
    
    /* Ensure the correct number of arguements */
    if (nrhs != 6)
        mexErrMsgTxt("Function requires 6 inputs");
    if (nlhs != 5)
        mexErrMsgTxt("Function requires 5 outputs");
    
    
    /* Assign pointers to the inputs */
    nr = mxGetScalar(prhs[0]);
    nc = mxGetScalar(prhs[1]);
    Cx = mxGetScalar(prhs[2]);
    Cy = mxGetScalar(prhs[3]);
    m = mxGetScalar(prhs[4]);
    l = mxGetScalar(prhs[5]);
    
    
    /* Set up the output matrices */
    plhs[0] = mxCreateDoubleMatrix(nr, nc, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(nr, nc, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(nr, nc, mxREAL);
    plhs[3] = mxCreateDoubleMatrix(2*nr-2, 2*nc-2, mxREAL);
    plhs[4] = mxCreateDoubleMatrix(1, 1, mxREAL);
    u_fish = mxGetPr(plhs[0]);
    v_fish = mxGetPr(plhs[1]);
    mask = mxGetPr(plhs[2]);
    mask_double = mxGetPr(plhs[3]);
    m_para = mxGetPr(plhs[4]);
    
    
    /* Call the main function */
    fish2para_coords(u_fish, v_fish, mask, mask_double, m_para, nr, nc, 
                                                            Cx, Cy, m, l);
}                     
    


/* 
 * Main Function
 */
int 
fish2para_coords(double *u_fish, double *v_fish, double *mask, 
                    double *mask_double, double *m_para, int nr, int nc, 
                    double Cx, double Cy, double m, double l)
{
    int i, j;
    int i2, j2;
    double x, y, x_sq, y_sq;
    double r, R;    
    double theta, phi;   
    double rescale;
    double mp;
    int nr2;
    
    /* 
     * Set the number of rows for the doubled image size
     * (need to index transpose coordinates so that the output is correct)
     */
    nr2 = 2*nr - 2;
    
    
    /* Find the scaling factor for the parabolic image model */
    mp = (l+m)/l - 1.0;
    *m_para = mp;
   
    
    /* 
     * For each pixel in the parabolic image, find the corresponding 
     * pixel position in the fisheye image 
     * (run columns in outer loop so that you do not need to do a 
     *  matrix transpose when converting back to matlab)
     */
    x = -1.0*Cx;
    for (j = 0; j < nc; j++) {
        x += 1.0;
        x_sq = x * x;
        j2 = j * 2;
        
        y = -1.0*Cy;
        for (i = 0; i < nr; i++, u_fish++, v_fish++, mask++) {
            y += 1.0;
            y_sq = y * y;
            i2 = i*2;
            
            /* Get radius on the image plane */
            R = sqrt(x_sq + y_sq);
     
            /* Find angle of longitude and colatitude on the sphere */
            theta = 2.0 * atan(R / (mp + 1.0));
            phi = atan2(y,x);
            
            /* 
             * If theta > pi/2, set the result to pi/2
             * (this stops a black border which has a bad effect when 
             *  finding the features => features on the border)
             */
            if (theta > 1.5708) {
                theta = 2.0*1.5708 - theta;
                *mask = 0;
                if ((i < nr-1) && (j < nc-1)) {
                    *(mask_double + j2*nr2 + i2) = 0;
                    *(mask_double + j2*nr2 + i2 +1) = 0;
                    *(mask_double + (j2+1)*nr2 + i2) = 0;
                    *(mask_double + (j2+1)*nr2 + i2 + 1) = 0;
                }
            } else {
                *mask = 1;
                if ((i < nr-1) && (j < nc-1)) {
                    *(mask_double + j2*nr2 + i2) = 1;
                    *(mask_double + j2*nr2 + i2 +1) = 1;
                    *(mask_double + (j2+1)*nr2 + i2) = 1;
                    *(mask_double + (j2+1)*nr2 + i2 + 1) = 1;
                }
            }
            
                
            /* Find the scaling factor to make the conversion */
            r = sin(theta);
            rescale = (l + m) / (l + cos(theta));
            
            /* Make the conversion and assign result to output */
            R = rescale * r;
            *u_fish = R * cos(phi) + Cx;
            *v_fish = R * sin(phi) + Cy;
        }
    }
    return 0;
}


