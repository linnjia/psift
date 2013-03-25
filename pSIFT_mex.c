/** 
 **  pSIFT_mex: Finds pSIFT keypoints
 **  See pSIFT.m and pSIFT_demo.m for example use.
 **
 **  Copyright (C) 2013 Peter Hansen [phansen.au(at)gmail.com]
 **
 **  
 ** This program is distributed in the hope that it will be useful,
 ** but WITHOUT ANY WARRANTY; without even the implied warranty of
 ** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 ** GNU General Public License for more details.
 **
 **/

 
#include "mex.h"
#include <math.h>
#include <stdio.h>
#define PI 3.14159265359
#define NMAX 6000           /* Hard coded upper limit on number keypoints */
#define indexbins 4
#define PWIDTH 20



/**
 ** Gateway function
 **=========================================================================*/
void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/* Outputs */
	double *key_tmp, *key;			/* Keypoints */
	
	/* Inputs */
	double *img;					/* Input image */
	double *mask_input;				/* Image mask input */
	int *mask;                      /* Interger copy */
	double *kernels;				/* Smoothing kernels */
	double *kernel_widths;			/* Half-widths of kernels */
	double *kt_scales;              /* The spherical scales kt */
	int noct, spo;					/* No. octaves and scales per pctave */
	double *mp;						/* 'm' values of the octave images */
	double DoG_thresh;				/* Diff. Gauss threshold */
	double Cx, Cy;					/* Principal point */
	
	/* Other */
    int nr, nc;                   	/* Original image size */
    int nc_kernels;                 /* Columns in kernels array */
    int *found;                	    /* Image yes/no features found */
    double *L, *D;                	/* Scale space and DoG images */
    double *buffer1, *buffer2;     	/* Convolution buffers */
    double *patch, *gmag, *gori;   	/* For descriptors */
    double *patch_tmp;             	/* Temp patch used for smoothing */
    int n;                        	/* Number of keypoints found */
    int i,j;

   
    /* Ensure the correct number of inputs and outputs */
    if (nrhs != 11)
        mexErrMsgTxt("Mex function requires 10 inputs");
    if (nlhs != 1)
        mexErrMsgTxt("Mex function requires 1 output");
    
    
    /* Get the inputs */
    img = mxGetPr(prhs[0]);
    mask_input = mxGetPr(prhs[1]);
    kernels = mxGetPr(prhs[2]);
	kernel_widths = mxGetPr(prhs[3]);
	kt_scales = mxGetPr(prhs[4]);
	noct = (int)(mxGetScalar(prhs[5]));
	spo = (int)(mxGetScalar(prhs[6]));
	mp = mxGetPr(prhs[7]);
	DoG_thresh = mxGetScalar(prhs[8]);
    Cx = mxGetScalar(prhs[9]) - 1.0;		/* Use C coords */
    Cy = mxGetScalar(prhs[10]) - 1.0;		/* Use C coords */
	
	
	/* The image size nr,nc (rows, cols), and kernel column count */
	nc = mxGetM(prhs[0]);
	nr = mxGetN(prhs[0]);
    nc_kernels = mxGetM(prhs[2]);
	
	
	/** Should really have some function here which checks sizes
	 ** e.g. mask and image same size, kernels correct sizes etc.
	 */
	
	
    /* Memory for scale-space images, found image, and mask */
	L = mxCalloc((2*nr-2) * (2*nc-2) * (spo + 3), sizeof(double));
	D = mxCalloc((2*nr-2) * (2*nc-2) * (spo + 2), sizeof(double));
	found = mxCalloc((2*nr-2) * (2*nc-2), sizeof(int));
    mask = mxCalloc((2*nr-2) * (2*nc-2), sizeof(int));
	
	/* Memory for convolution buffers */
	buffer1 = mxCalloc(4*nc, sizeof(double));
	buffer2 = mxCalloc(4*nc, sizeof(double));

	/* Memory for params used to evaluate descriptors */
	patch = mxCalloc((2*PWIDTH+1) * (2*PWIDTH+1), sizeof(double));
    gmag = mxCalloc((2*PWIDTH+1) * (2*PWIDTH+1), sizeof(double));
    gori = mxCalloc((2*PWIDTH+1) * (2*PWIDTH+1), sizeof(double));
    patch_tmp = mxCalloc((2*PWIDTH + 1) * (2*PWIDTH + 1), sizeof(double));
    
    /* Memory for keypoints (up to max number) */
    key_tmp = mxCalloc(NMAX * (4 + 128), sizeof(double));
    
    
	
    /**
	 ** Call the main pSIFT function and return the number of keypoints
	 **/
    n = pSIFT(key_tmp, img, mask_input, mask, kernels, kernel_widths, kt_scales, 
              noct, spo, mp, DoG_thresh, Cx, Cy, L, D, found, buffer1, 
              buffer2, patch, patch_tmp, gmag, gori, nr, nc, nc_kernels); 
							
              
     /* Set up the output keypoints */
    if (n > 0) {
        plhs[0] = mxCreateDoubleMatrix(132,n,mxREAL);
         key = mxGetPr(plhs[0]);
   
         /* Convert the results to the output */
         for (i = 0; i < n; i++) {
             for (j = 0; j < 132; j++) {
                 *(key + i*132 + j) = *(key_tmp + i*132 + j);
             }
         }
     } else {
         plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
         key = mxGetPr(plhs[0]);
         *key = -1;
     }
    
       
    /* Clean up */
    mxFree(L);
    mxFree(D);
    mxFree(found);
    mxFree(mask);
    mxFree(buffer1);
    mxFree(buffer2);
    mxFree(patch);
    mxFree(gmag);
    mxFree(gori);
    mxFree(patch_tmp);
    mxFree(key_tmp);
}                     
    



/** 
 ** Main Function
 **
 *==========================================================================*/
int
pSIFT(double *key_tmp, double *img, double *mask_input, int *mask, 
      double *kernels, double *kernel_widths, double *kt_scales, int noct, 
      int spo, double *mp, double DoG_thresh, double Cx, double Cy, 
      double *L, double *D, int *found, double *buffer1, double *buffer2, 
      double *patch, double *patch_tmp, double *gmag, double *gori, 
      int nr, int nc, int nc_kernels) 
{
  
    int i,j;
    int octave;
    int nfeatures;
    int *found_ptr;
    double m_octave;
    double *kt_octave;
    int n_init;
    
   
    /* Double mask (converts from original double to int) */
    double_mask(mask, mask_input, nr, nc);

    
    /* Set the map 'found' of yes/no features at pixel position */
    found_ptr = &found[0];
    for (i = 0; i < n_init; i++, found_ptr++) {
        *found_ptr = 0;
    }
    
    
    /* 
     * Double the image.
     * Stores the double image at second position in L (wrt. double size)
     * i.e. L[nr_double * nc_double]
     * Returns uptaded image sizes and principal point
     */
    double_image(L, img, &Cx, &Cy, &nr, &nc);
    n_init = nr * nc;
    
    /* Pre-smooth the image */
    octave = 1;
    diffuse_image(L, &L[nr*nc], buffer1, buffer2, octave, 1, spo, 
                        kernels, kernel_widths, nr, nc, nc_kernels);
    
                        
    
    /* 
     * For each octave, find the scale space and difference of Guassian
     * images from which features and descriptors are found
     */
    nfeatures = 0;
    while (octave <= noct) {

        
        /* The stereographic distance for this octave */
        m_octave = *(mp + octave - 1);
        
        /* Point to the start scale for this octave */
        kt_octave = &kt_scales[(octave-1) * (spo+3)];
        
        
        /* Find all the scale space images */
        for (i = 1; i < spo+3; i++) {
            diffuse_image(&L[nr*nc*i], &L[nr*nc*(i-1)], buffer1, buffer2, 
                octave, i+1, spo, kernels, kernel_widths, nr, nc, nc_kernels); 
        }
        
        /* Find the difference of Gaussian images */
        for (i = 1; i <= spo+2; i++) {
            subtract_image(&D[nr*nc*(i-1)],&L[nr*nc*(i-1)],&L[nr*nc*i],nr,nc);
        }
        
        
        /* For each valid scale, find the keypoints and descriptors */
        for (i = 1; i <= spo; i++) {
            get_keypoints(&nfeatures, key_tmp, &D[nr*nc*(i-1)], &D[nr*nc*i],
                    &D[nr*nc*(i+1)], &L[nr*nc*i], mask, found, patch, 
                    patch_tmp, gmag, gori, Cx, Cy, m_octave, octave, i, 
                    kt_octave, DoG_thresh, nr, nc);
        }
        
           
        /* 
         * Halve the image mask, found image, and image.
         * The halve mask, found have to be called first because
         * halve image changes the nr,nc values.
         */
        if (octave < noct) {
            halve_mask(mask, nr, nc);
            halve_mask(found, nr, nc);
            halve_image(L, &L[nr*nc*(spo+1)], &Cx, &Cy, &nr, &nc);
        }
        octave++;
    }     
    
    /* Return the number of features found */
    return nfeatures;
}
    
    
    



/* 
 * Gets the keypoints given the input difference of Gaussian images
 */
int
get_keypoints(int *nfeatures, double *key_tmp, double *D1, double *D2, 
            double *D3, double *im, int *mask, int *found, double *patch, 
            double *patch_tmp, double *gmag, double *gori, double Cx, double Cy, 
            double m, int octave, int scale_index, double *kt_octave, 
            double DoG_thresh, int nr, int nc)
{
    int i, j;
    int n, n1, n2, n3;
    int nfeat_tmp;
    double val;
    double u, v, scale;         /* Keypoint position on image and scale */


    /* Find all the scale invariant features and descriptors */
    n1 = 0;
    n2 = 0;
    n3 = 0;
    nfeat_tmp = *nfeatures;
    for (i = 5; i < nr-5; i++) {
        for (j = 5; j < nc-5; j++) {
            
            /* Ensure the point is within the mask */
            if (*(mask + i*nc + j) == 0) {
                continue;
            } 
                
            /* 
             * Test that value is above thhreshold, a local extrema, 
             * and not on an edge 
             */
            val = *(D2 + i*nc + j);
            if ((fabs(val) > 0.8 * DoG_thresh) &&
                (ismaxmin(val, D1, i, j, nc) == 1) && 
                (ismaxmin(val, D2, i, j, nc) == 1) &&
                (ismaxmin(val, D3, i, j, nc) == 1)) {  
                n1++;
                        
                if (not_edge(D2, i, j, nc) == 1) {
                    n2++;
                        
                    /* Interpolate the keypoint */
                    if (interpolate_keypoint(&u, &v, &scale, D1, D2, D3, 
                            found, scale_index, i, j, nr, nc, 
                            DoG_thresh, Cx, Cy, m, octave, kt_octave) == 1) {       
                        n3++;
                        
                    
                        /* Get the descriptor and store position */
                        n = spherical_SIFT_descriptor(key_tmp, im, patch, 
                            patch_tmp, gmag, gori, u, v, scale, Cx, Cy, 
                            m, nr, nc, nfeat_tmp, octave);
     
                        nfeat_tmp +=n;
                    }
                }
            }
        }
    }
    
    /* Update the feature count */
    *nfeatures = nfeat_tmp;
    
    return 0;
}



 
/*
 * Test to see if the point is a local minima or maxima
 * (Use 3x3 or 5x5 ???)
 */
int
ismaxmin(double val, double *D, int r, int c, int nc)
{
    int i, j;
    int w = 1;
    
    if (val < 0) {
        for (i = r-w; i <= r+w; i++) {
            for (j = c-w; j <= c+w; j++) { 
                if (*(D + i*nc + j) < val) {
                    return 0;
                }
            }
        }
    } else {
        for (i = r-w; i <= r+w; i++) {
            for (j = c-w; j <= c+w; j++) { 
                if (*(D + i*nc + j) > val) {
                    return 0;
                }
            }
        }
    }
    return 1;
}




/* 
 * Test to see if the point is on an edge
 * Remember that Hessian is symmetric (hxy = hyx)
 */
int
not_edge(double *D, int i, int j, int nc) 
{
    double H_11, H_12, H_22;
    double detH, trH;
    double ratio = 10.0;    /* 10.0 */
    double val1, val2;
    
    /* Get the Hessian matrix */
    H_22 = *(D + i*nc + (j+1)) + *(D + i*nc + (j-1)) - 
                2.0 * *(D + i*nc + j);
    H_11 = *(D + (i+1) * nc+j) + *(D + (i-1)*nc + j) - 2.0 * *(D+i*nc+j);
    H_12 = ((*(D+(i+1)*nc+(j+1)) - *(D+(i+1)*nc+(j-1))) - 
            (*(D+(i-1)*nc+(j+1)) - *(D+(i-1)*nc+(j-1)))) / 4.0;
    
    /* Find the determinant and trace */
    detH = (H_11 * H_22) - (H_12 * H_12);
    trH = H_11 + H_22;
        
    /* See if the extrema is an edge response */
    val1 = detH * (ratio + 1.0) * (ratio + 1.0);
    val2 = ratio * trH * trH;
    if (val1 > val2) {
        return 1;
    } else {
        return 0;
    }
}
   





/* 
 * Interpolate position and scale of keypoint using 3D parabolic fit
 */
int
interpolate_keypoint(double *u, double *v, double *scale, double *D1, 
                double *D2, double *D3, int *found, int scale_index, 
                int i, int j, int nr, int nc, double DoG_thresh, 
                double Cx, double Cy, double m, int octave, double *kt_octave)
{
    int trial; 
    double D[3];
    double H[3][3];
    double offset[3];
    double *offset_ptr;
    int change;
    double sign;
    double val;
    /*double R_sq;*/
    double scale_offset;
    
    
    offset_ptr = offset;
    
    for (trial = 1; trial <= 5; trial++) {
        
        change = 0;
        
        /* Gradient */
        D[0] = (*(D2 + i*nc + j+1) - *(D2 + i*nc + j-1)) / 2.0;
        D[1] = (*(D2 + (i+1)*nc + j) - *(D2 + (i-1)*nc + j)) / 2.0;  
        D[2] = (*(D3 + i*nc + j) - *(D1 + i*nc + j)) / 2.0; 
    
        /* Hessian */
        H[0][0] = *(D2 + i*nc + j+1) - 2.0 * *(D2 + i*nc + j) + 
                    *(D2 + i*nc + j-1);                
        H[1][1] = *(D2 + (i+1)*nc + j) - 2.0 * *(D2 + i*nc + j) + 
                    *(D2 + (i-1)*nc + j);
        H[2][2] = *(D3 + i*nc + j) - 2.0 * *(D2 + i*nc + j) + 
                    *(D1 + i*nc + j);
    
        H[0][1] = ((*(D2 + (i+1)*nc + j+1) - *(D2 + (i+1)*nc + j-1)) - 
                (*(D2 + (i-1)*nc + j+1) - *(D2 + (i-1)*nc + j-1))) / 4.0;
    
        H[0][2] = ((*(D3 + i*nc + j+1) - *(D3 + i*nc + j-1)) -
                   (*(D1 + i*nc + j+1) - *(D1 + i*nc + j-1))) / 4.0;
   
        H[1][2] = ((*(D3 + (i+1)*nc + j) - *(D3 + (i-1)*nc + j)) - 
                   (*(D1 + (i+1)*nc + j) - *(D1 + (i-1)*nc + j))) / 4.0;
    
        H[1][0] = H[0][1];
        H[2][0] = H[0][2];
        H[2][1] = H[1][2];
    
    
        /* The interpoolated values are the solution of the linear system
         * Hx = -D  => x = - inv(H)*D
         */
        solve_gauss_elimination(offset, D, H);
    
        
        /* Increment the i,j values based on the offset */
        if (fabs(offset[0]) > 0.6) {
            if (offset[0] > 0.0) {
                j += 1;
            } else {
                j -= 1;
            }
            change = 1;
        }
        
        if (fabs(offset[1]) > 0.6) {
            if (offset[1] > 0.0) {
                i += 1;
            } else {
                i -= 1;
            }
            change = 1;
        }
   
        
        /* Terminate loop if there is no change in position or no longer 
         * within the image
         */
        if ((change == 0) || (i<=1) || (i>=nr-2) || (j<=1) || (j>=nc-2)) {
            break;
        }
    }
    
    /* Only accept the new position and scale if the offset is not too
     * large, and not on a position already taken */
    if ((fabs(offset[0]) < 0.8) && (fabs(offset[1]) < 0.8) && 
        (fabs(offset[2]) < 0.75) && 
        (*(found + (int)(i+offset[0])*nc + (int)(j+offset[2])) == 0)) {
 
        /* Find the new interpolated DoG value */
        val = *(D2 + i*nc + j) + 0.5*(offset[0]*D[0] + offset[1]*D[1] +
                                      offset[2]*D[2]);
       
        /* Check the new DoG value */
        if (fabs(val) > DoG_thresh) {    
            
            /* Get the new keypoint position for this octave */
            *u = (double)(j) + offset[0];
            *v = (double)(i) + offset[1];
            /*R_sq = pow(*u - Cx, 2.0) + pow(*v - Cy, 2.0);*/
            
            
            /*
             * Get the keypoint scale => solved by considering the sphere 
             * and image planes as smooth manifolds.
             */
            /*spherical_scale(scale, scale_index, offset[2], 
                                            R_sq, m, octave, scale_mode);*/
            if (offset[2] < 0) {
                scale_offset = offset[2] * (sqrt(*(kt_octave + scale_index)) - 
                                            sqrt(*(kt_octave + scale_index-1)));
            } else {
                scale_offset = offset[2] * (sqrt(*(kt_octave + scale_index+1)) - 
                                            sqrt(*(kt_octave + scale_index)));
            }
            *scale = sqrt(*(kt_octave + scale_index)) + scale_offset;
            
            
            /* Update the map */
            *(found + (int)(i+offset[0])*nc + (int)(j+offset[1])) = 1;
            
            return 1;
        } else {
            return 0;
        }
    } else {
        return 0;
    }
    return 0;
}
     

        
        

/*
 * Given a 3x3 linear system, solves Hx = -D using Gauss elimination
 * Pivoting of the rows is used to prevent division by small numbers 
 */
int
solve_gauss_elimination(double *offset, double D[3], double H[3][3])
{
    int i;
    int row, col;           /* Outer loop row iteration */
    int c;                  /* Row column index */
    int piv_row;            /* Pivot row index */
    double maxval, val;     /* Temporary values */
    double Htemp, Dtemp;
    double Sval;            /* Coefficients for division and subtraction */
    double Ou, Ov, Os;      /* Offsets -> u, v, scale */
    double negD[3];
    
    /* Copy contents over to avoid confusion */
    for (i = 0; i < 3; i++) {
        negD[i] = -D[i];
    }
    
   
    /* 
     * Reduce matrix H to upper triangle by Gauss elimination
     * Only rows are pivoted as this does not change the order
     * of the final solution 
     */
     for (col = 0; col < 2; col++) {
        
        /* Find the pivot row */
        maxval = fabs(H[col][col]);
        piv_row = col;
        for (row = col+1; row < 3; row++) {
            val = fabs(H[row][col]);
            if (val > maxval) {
                maxval = val;
                piv_row = row;
            }
        }
        
        /* 
         * Pivot the largest value to the top
         * (Exchange the values in the matrix H and D) 
         */
        if (piv_row != col) {
            Dtemp = negD[piv_row];
            negD[piv_row] = negD[col];
            negD[col] = Dtemp;
            for (i = 0; i < 3; i++) {
                Htemp = H[piv_row][i];
                H[piv_row][i] = H[col][i];
                H[col][i] = Htemp;
            }
        }
           
        /*
         * Divide throught each row (make diagonal 1) and then reduce
         * the remaining columns by subtraction 
         * nb. You do not HAVE to actually DO the division to make 
         * diagonal 1, as long as you use the correct multiplication 
         * when doing the subtractions.
         */
        for (row = col+1; row < 3; row++) {
            Sval = H[row][col] / H[col][col];
            negD[row] -= Sval * negD[col];
            for (c = col; c < 3; c++) {
                H[row][c] -= Sval * H[col][c];
            }
        }
    } 
            
    /* Solve via back substitution */
    Os = negD[2]/H[2][2];
    Ov = (negD[1] - H[1][2]*Os) / H[1][1];
    Ou = (negD[0] - H[0][2]*Os - H[0][1]*Ov) / H[0][0];
            
    offset[0] = Ou;
    offset[1] = Ov;
    offset[2] = Os;
    
    return 1;
}






/* NOT USED: NOW INPUT KT VALUES INTO PSIFT
 * Finds the keypoint scale on the sphere
 * (corrects scale based om tensor between sphere and parabolic image plane)
 *--------------------------------------------------------------------------*/
int
spherical_scale(double *scale, int scale_index, double offset, 
                double R_sq, double m, int octave, int scale_mode)
{
    double m1_sq;
    double dR_sq;
    double kt1, kt2;
    double d_scale_index = (double)(scale_index);
    double doctave = (double)(octave);
    double scale_init;
    double sqrt_kt_init;
    double SQRT_KT_INIT_PARA = 1;
    double SQRT_KT_INIT_FISH = 1;
    double dSPO = 1;
    
    /* Correct initial scale for the current octave */
    sqrt_kt_init = (scale_mode == 0 ? SQRT_KT_INIT_PARA : SQRT_KT_INIT_FISH);
    scale_init = sqrt_kt_init * pow(2.0, doctave - 1.0);
    
    /* Add the offset value */
    kt1 = pow(scale_init * pow(2.0, (d_scale_index + offset)/dSPO), 2.0);
  
    /* Correct scale for the given radius from the image centre */
    m1_sq = pow((m + 1.0), 2.0);
    kt2 = kt1 * pow(m1_sq, 2.0) / pow(m1_sq + R_sq, 2.0);
    
    *scale = sqrt(kt2);
    
    return 0;
}





/* 
 * Subtracts two images 
 *--------------------------------------------------------------------------*/
int
subtract_image(double *output, double *im1, double *im2, int nr, int nc)
{
    int i;
    int imax;
    
    imax = nr*nc;
    
    for (i = 0; i < imax; i++, output++, im1++, im2++) {
        *output = *im2 - *im1;
    }
    
    return 0;
}
    



/* 
 * Doubles the original mask (converting from double to int)
 *--------------------------------------------------------------------------*/
int 
double_mask(int *output, double *mask, int nr, int nc)
{
    int r1, c1, r2, c2;
    int nr_new, nc_new;
    double *maskptr;
    int val;
    int *outputptr;
    
    /* Set the new image size */
    nr_new = nr * 2 - 2;
    nc_new = nc * 2 - 2;
    
    /* Double the mask */
    for (r1 = 0; r1 < nr-1; r1++) {
        r2 = 2*r1;
        for (c1 = 0; c1 < nc-1; c1++) {
            c2 = 2*c1;
            maskptr = &mask[c1 + r1 * nc];
 
            /* Output ptr */
            outputptr = &output[c2 + r2 * nc_new];
   
            if (*maskptr > 0.0) {
                val = 1;
            } else {
                val = 0;
            }
                  
            *outputptr = val;
            *(outputptr + 1) = val;
            *(outputptr + nc_new) = val;
            *(outputptr + nc_new + 1) = val;
        }
    }
    
    return 0;
}






/* 
 * Doubles the original input image size
 *--------------------------------------------------------------------------*/
int 
double_image(double *output, double *im, double *Cx, double *Cy, int *nr, int *nc)
{
    int r1, c1, r2, c2;
    int nr_new, nc_new;
    double *ptr1, *ptr2;
    
    /* Set the new image size */
    nr_new = *nr * 2 - 2;
    nc_new = *nc * 2 - 2;
    
    /* Double the image */
    for (r1 = 0; r1 < *nr-1; r1++) {
        r2 = 2*r1;
        for (c1 = 0; c1 < *nc-1; c1++) {
            c2 = 2*c1;
            ptr1 = &im[c1 + r1 * *nc];
 
            /* Remember that output is stored effectively in L(:,:,2) */
            ptr2 = &output[c2 + r2 * nc_new + (nr_new * nc_new)];
   
            *ptr2 = *ptr1;
            *(ptr2 + 1) = 0.5 * (*ptr1 + *(ptr1 + 1));
            *(ptr2 + nc_new) = 0.5 * (*ptr1 + *(ptr1 + *nc));
            *(ptr2 + nc_new + 1) = 0.25 * 
                (*ptr1 + *(ptr1 + 1) + *(ptr1 + *nc) + *(ptr1 + *nc + 1));
        }
    }
    
    /* Update the camera model values and the image size */
    *Cx *= 2.0;
    *Cy *= 2.0;
    /* *m = (2.0 * (*m + 1.0)) - 1.0;*/
    *nr = nr_new;
    *nc = nc_new;
    
    return 0;
}





/*
 * Halves the image size 
 *--------------------------------------------------------------------------*/
int
halve_image(double *output, double *input, double *Cx, double *Cy, 
                                                        int *nr, int *nc)
{
    int r, c;
    int nr_new, nc_new;
    double *ptr1;
    
    /* Set the new image size */
    nr_new = *nr / 2;
    nc_new = *nc / 2;
    
    /* Halve the image (sampling every second pixel) */
    for (r = 0; r < nr_new; r++) {
        ptr1 = &input[r*2 * *nc];
        for (c = 0; c < nc_new; c++, output++, ptr1+=2) {
            *output = *ptr1;
        }
    }
       
    /* Update the camera model values and the image size */
    *Cx /= 2.0;
    *Cy /= 2.0;
    /* *m = ((*m + 1.0) / 2.0) - 1.0; */
    *nr = nr_new;
    *nc = nc_new;
    
    return 0;
}




/*
 * Halves the mask image
 * Do not require a 'buffer' image, simply overwrites values which is
 * OK as the image size is halved.
 *--------------------------------------------------------------------------*/
int
halve_mask(int *mask, int nr, int nc)
{
    int r, c;
    int nr_new, nc_new;
    int *ptr_mask = mask;
    int *ptr_mask2;
    
    /* Set the new image size */
    nr_new = nr / 2;
    nc_new = nc / 2;
    
    /* Halve the mask (sampling every second pixel) */
    for (r = 0; r < nr_new; r++) {
        ptr_mask2 = &mask[r*2 * nc];
        for (c = 0; c < nc_new; c++, ptr_mask++, ptr_mask2+=2) {
            *ptr_mask = *ptr_mask2;
        }
    }
    return 0;
}








/* 
 * Filters the image using the spherical diffusion function mapped to 
 * the image plane.  Requires the octave and scale to be given so that the 
 * correct kernel can be selected.
 *
 * !!! Assumes the kernels have rank 1 (ie. kernel is perfectly
 *      separable).  Could find best rank 1 estimate through SVD, but
 *      for simplicity the 1D kernels used are just the middle row
 *      and column of the full kernel.
 */
int
diffuse_image(double *output, double *input, double *buffer1, 
            double *buffer2, int octave, int scale, int spo, double *kernels, 
            double *kernel_widths, int nr, int nc, int nc_kernels)
{
    int width;                  /* Kernel width */
    double *kernelsptr;          /* Pointer to start value in kernel */
    int kernel_index;           /* Which number kernel */
    double *imptr;
    double *outptr;
    int r, c;
    int i;
    
    /* The kernel index */
    if ((octave == 1) && (scale == 1)) {
        kernel_index = 0;
    } else {
        kernel_index = (octave-1) * (spo+3) + scale;
    }
        
    
    /* Get kernel width and set ptr. to start position */
    width = *(kernel_widths + kernel_index);
    kernelsptr = &kernels[kernel_index * nc_kernels];
    
    
    /* Do horizonal convolution */
    imptr = &input[0];
    outptr = &output[0];
    for (r = 0; r < nr; r++) {
        
        /* Pad the row and put into buffer1 */
        for (i = 0; i < width; i++) {
            *(buffer1 + i) = *imptr;
        }
        for (i = 0; i < nc; i++, imptr++) {
            *(buffer1 + width + i) = *imptr;
        }
        for (i = 0; i < width; i++) {
            *(buffer1 + width + nc + i) = *(imptr-1);
        }
        
        /* Do the convolution */
        buffer_convolution(buffer2, buffer1, kernelsptr, width, nc);
        
        /* Update the result */
        for (i = 0; i < nc; i++, outptr++) {
            *outptr = *(buffer2 + i);
        }
    }
        
    
    /* Do vertical convolution */
    for (c = 0; c < nc; c++) {
        
        imptr = &output[c];
        
        /* Pad the row and put into buffer1 */
        for (i = 0; i < width; i++) {
            *(buffer1 + i) = *imptr;
        }
        for (i = 0; i < nr; i++, imptr += nc) {
            *(buffer1 + width + i) = *imptr;
        }
        for (i = 0; i < width; i++) {
            *(buffer1 + width + nr + i) = *(imptr-nc);
        }
        
        /* Do the convolution */
        buffer_convolution(buffer2, buffer1, kernelsptr, width, nr);
        
        /* Update the result */
        outptr = &output[c];
        for (i = 0; i < nr; i++, outptr+=nc) {
            *outptr = *(buffer2 + i);
        }
    }
    return 0;
}





/* 
 * Returns the result of convolution given the buffer and kernel 
 *--------------------------------------------------------------------------*/
int 
buffer_convolution(double *buffer2, double *buffer1, 
                                        double *kernel, int width, int n)
{
    int i, j;
    double sum;
    double *kptr;
    double *b1_ptr;
    int flag = 0;
    
    kptr = &kernel[0];
    b1_ptr = &buffer1[width];
    for (i = 0; i < n; i++, buffer2++, b1_ptr++) {
                            
        /* Get the result */
        sum = *kptr * *b1_ptr;
        for (j = 1; j <= width; j++) {
            sum += *(kptr+j) * (*(b1_ptr-j) + *(b1_ptr+j));
        }
        *buffer2 = sum;
    }
    return 0;
}








/* 
 *  DESCRIPTORS CODE
 *======================================================================*/

/*
 * Samples region to fixed size patch and then finds the SIFT descriptors
 * => writes the result to the key_tmp list and returns the number of 
 *    keypoints (can have multiple oritentations)
 */
int
spherical_SIFT_descriptor(double *key_tmp, double *im, double *patch, 
        double *patch_tmp, double *gmag, double *gori, double u, 
        double v, double scale, double Cx, double Cy, double m, 
        int nr, int nc, int nfeat, int octave)
{
    int i, j, k;
    double psi_support;     /* Angle of support region on sphere */
    int n_ori;
    double indexscale;
    double n;
    double ori[36];
   
    
    /*
     * For the given patch width, find the equivalent indexscale
     * for the patch 
     */
    indexscale = ceil((2.0 * (double)(PWIDTH)) / (5.0 * sqrt(2.0)));
   
    
    /* 
     * Map the region on the image defined by the 
     * scale to the fixed sized patch and then find descriptor
     * (remember to rlationship =>  psi = sqrt(2) * sqrt(kt), where 
     *  the value of 'scale' is sqrt(kt))
     */ 
    psi_support = (sqrt(2.0) * scale) * (sqrt(2.0) * 3.0 * 5.0 / 2.0);
    
    sample_image_equiangular(patch, im, u, v, psi_support, Cx, Cy, m, nr, nc);

    
    /* Do convolution with Gaussian */
    /*smooth_patch(patch, patch_tmp);*/
             
     /* Test that the keypoint doesn't lie on an edge
      * Now have better approximation of undistorted region 
      */
    if (not_edge_patch(patch) == 1) {
    
        /* 
         * Find the gradient magnitude and orientation for all 
         * pixels in the patch not on the border
         */
        gradient_ori_mag(gori, gmag, patch);
        
        
        /* Find the keypoint orientaton(s) */
        n_ori = keypoint_orientation(&ori, gori, gmag);
           
    
        /* Get the SIFT descriptor for each orientation */
        for (j = 0; j < n_ori; j++) {
            *(key_tmp + nfeat*132) = pow(2.0,octave-1.0) * (u / 2.0) + 1.0;
            *(key_tmp + nfeat*132 + 1) = pow(2.0,octave-1.0) * (v / 2.0) + 1.0;
            *(key_tmp + nfeat*132 + 2) = scale;
            *(key_tmp + nfeat*132 + 3) = *(ori + j);
       
            /* 
             * Get the SIFT descriptor 
             * (send pointer to key_tmp where it should start writing to)
             */
            keypoint_descriptor(&key_tmp[nfeat*132 + 4], gori, gmag,
                                                    *(ori + j), indexscale);
            nfeat++;
        }
        return n_ori;
    } else {
        return 0;
    }
}





/*
 * Samples the image for the given keypoint position and scale 
 */
int 
sample_image_equiangular(double *patch, double *im, double u, double v, 
                             double psi, double Cx, double Cy, double m,
                             int nr, int nc)
{
    double sx, sy, sz;      /* Point on the sphere */
    double Sx, Sy, Sz;      /* Point on the sphere (keypoint at pole) */
    double beta, gamma;     /* y,z euler rotation angles */
    double Cbeta, Sbeta, Cgamma, Sgamma;    /* sin/cos of angles */
    double R[3][3];         /* Rotation matrix */
    double scaling;
    double theta, phi;      /* Spherical angles */
    double Rpatch;          /* Radius on the patch */
    double U, V;
    int i,j;
    double diff_u, diff_v;
    int U_int, V_int;
    double x,y;
    double l = 1.0;
    int Pw = (int)(PWIDTH);
    
     /* Find the x,z Euler angles required for rotation */
    if (map_stereographic_img2sphere(&sx, &sy, &sz, u-Cx, v-Cy, m) == 1) {
        gamma = atan2(-sy,sx);
        beta = -acos(sz);
    } else {
        return 1;
    }
        
    /* Set up the rotation matrix (map from the pole back to original) */
    Cbeta = cos(beta);
    Sbeta = sin(beta);
    Cgamma = cos(gamma);
    Sgamma = sin(gamma);
    R[0][0] = Cbeta * Cgamma;
    R[0][1] = Sgamma;
    R[0][2] = -Sbeta * Cgamma;
    R[1][0] = -Cbeta * Sgamma;
    R[1][1] = Cgamma;
    R[1][2] = Sbeta * Sgamma;
    R[2][0] = Sbeta;
    R[2][1] = 0.0;
    R[2][2] = Cbeta;
       
    
    /* Find the required scaling value so that the edge pixel 
     * in the patch maps to the support radius */
    scaling =  psi / (double)(PWIDTH) ;

    /* 
     * For each point on the patch, find the position on the sphere, 
     * rotate to the new point and then project to the image plane
     * and sample the value
     */
    for (i = -Pw; i <= Pw; i++) {
        y = (double)(i);
        
        for (j = -Pw; j <= Pw; j++, patch++) {
            x = (double)(j);
            
            /* Get position on sphere wrt pole */
            theta = scaling * sqrt(x*x + y*y);
            phi = atan2(y,x);
                
            Sx = sin(theta)*cos(phi);
            Sy = -sin(theta)*sin(phi);
            Sz = cos(theta);
        
            /* Rotate to the new point */
            sx = Sx*R[0][0] + Sy*R[0][1] + Sz*R[0][2];
            sy = Sx*R[1][0] + Sy*R[1][1] + Sz*R[1][2];
            sz = Sx*R[2][0] + Sy*R[2][1] + Sz*R[2][2];
        
            /* Project to the image plane and subtract 1 for C coords */
            U = (sx*(l+m) / (l+sz)) + Cx;
            V = (sy*(l+m) / (l+sz)) + Cy;
            
            /* Do linear interpolation of the image value */
            if ((U > 0.0) && (U < (double)(nc-1.0)) && 
                (V > 0.0) && (V < (double)(nr-1.0))) {
            
                U_int = (int)(floor(U));
                V_int = (int)(floor(V));
                diff_u = U - (double)(U_int);
                diff_v = V - (double)(V_int);
               
                *patch = *(im + V_int*nc + U_int) * (1.0-diff_u)*(1.0-diff_v) +
                    *(im + V_int*nc + U_int+1) * (diff_u)*(1.0-diff_v) +
                    *(im + (V_int+1)*nc + U_int) * (1.0-diff_u)*(diff_v) + 
                    *(im + (V_int+1)*nc + U_int+1) * (diff_u)*(diff_v);
            } else {
                *patch = 0.0;
            }
        }
    }
    return 0;
}






/* 
 * Map a point on the image plane to the unit radius viewing sphere 
 * Returns 1 if a solution exists, otherwise 0 
 * Possible imaginary solutions for l > 1 (no circle/line interection)
 */
int 
map_stereographic_img2sphere(double *sx, double *sy, double *sz,
               double x, double y, double m)
{
    double R, phi;
    double theta, r;
    
    if ((x == 0.0) && (y == 0.0)) {
        *sx = 0.0;
        *sy = 0.0;
        *sz = 1.0;
        return 1;
    }
    
    R = sqrt(x*x + y*y);
    phi = atan2(y,x);
    theta = 2.0 * atan(R / (m + 1.0));
    r = sin(theta);
    
    *sx = r * cos(phi);
    *sy = r * sin(phi);
    *sz = cos(theta);
    
    return 1;
}
    
   
        


/*
 * For the given patch, find the magnitude of the gradient and the
 * gradient orientation 
 */
int
gradient_ori_mag(double *gori, double *gmag, double *patch)
{
    int i, j;
    double dx, dy;
    int Pw = (int)(PWIDTH);
    int nr = 2*Pw+1;
        
    for (i = -Pw; i <= Pw; i++) {
        for (j = -Pw; j <= Pw; j++, gori++, gmag++, patch++) {
            
            if ((i == -Pw) || (j == -Pw) || (i == Pw) || (j == Pw)) {
                *gori = 0.0;
                *gmag = 0.0;
            } else {
            
            	/* May consider 3x3 kernel */
                dx = *(patch+1) - *(patch-1);
                dy = *(patch+nr) - *(patch-nr);
                 
                *gori = atan2(dy,dx);
                *gmag = sqrt(dx*dx + dy*dy);
            }
        }
    }
    return 0;
}







/*
 * Finds the keypoint orientations
 */
int
keypoint_orientation(double *ori, double *gori, double *gmag)
{
    int i, j, k;
    double hist[36];        /* Gradient orientation histogram */
    double dhistbin;        /* Double value of histogram bin */
    int histbin[2];         /* Histogram bins */
    double histweight[2];   /* Histogram bin weights (interpolation) */
    double histdiff;
    double distsq, weight;
    double mag;
    double first, middle, last;
    double maxval;
    int bindiff;
    double offset;
    int Pw = (int)(PWIDTH);
    int Pw_ori;
    double sigma_ori;  /* Used to weight the values */
    double ori_tmp;
    double x,y;
    
    /* Ensure all values in the histogram are zero */
    for (i = 0; i < 36; i++) {
        hist[i] = 0.0;
    }
    
    Pw_ori = Pw - 1;
    sigma_ori = (double)(PWIDTH) / 1.5;
    for (i = -Pw_ori; i <= Pw_ori; i++) {
        y = (double)(i);
        
        for (j = -Pw_ori; j <= Pw_ori; j++, gori++, gmag++) {
            x = (double)(j);
            
            /* Get the histogram bin */
            dhistbin = 36.0 * (*gori + PI + 0.001) / (2.0 * PI);
            
            /* Set this as integer histogram bins and find weights */
            histbin[0] = (int)(floor(dhistbin));
            histbin[1] = histbin[0] + 1;
            histdiff = dhistbin - (double)(histbin[0]);
            histweight[0] = 1.0 - histdiff;
            histweight[1] = histdiff;
                
            /* Do histogram bin wrap around if required */
            if (histbin[0] > 35) {
                histbin[0] -= 36;
            }
            if (histbin[1] > 35) {
                histbin[1] -= 36;
            }
                
            /* Get the gradient magnitude */
            mag = *gmag; 
                
            /* Find the weighting for the gradient magnitude */
            distsq = (x*x + y*y);
            weight = exp(-distsq / (2.0 * sigma_ori * sigma_ori));
            mag *= weight;
                
            /* Now add the values to the histogram */
            hist[histbin[0]] += mag * histweight[0];
            hist[histbin[1]] += mag * histweight[1];
        }   
    }
    
    /* Smooth the histogram bins to improve accuracy of results */
    for (i = 0; i < 6; i++) {
        first = hist[35];
        for (j = 0; j < 36; j++) {
            middle = hist[j];
            last = hist[(j+1) == 36 ? 0 : j+1];
            hist[j] = (first + middle + last) / 3.0;
            first = middle;
        }
    }
     
    
    /* Find the largest value in the histogram */
    maxval = hist[0];
    for (i = 1; i < 36; i++) {
        if (hist[i] > maxval) {
            maxval = hist[i];
        }
    }
    
    /* 
     * Look for all local maxima in the histogram.  If the value is
     * within 0.8 of maxval, interpolate orientation
     */ 
    k = 0;
    for (i = 0; i < 36; i++) {
        first = hist[(i == 0 ? 35 : i-1)];
        last = hist[(i == 35 ? 0 : i+1)];
        
        if ((hist[i] > first) && (hist[i] > last) && 
            (hist[i] > 0.8 * maxval)) {
                
            /* Interpolate the orientation (parabolic fit) */
            interpolate_parabola(&offset, first, hist[i], last);    
    
            /* Now convert back to an angle (0 to 2pi) */
            ori_tmp = ((double)(i) + offset + 0.5) * (2.0 * PI / 36.0) - PI;
            while (ori_tmp < PI) {
                ori_tmp += 2.0*PI;
            }
            while (ori_tmp > PI) {
                ori_tmp -= 2.0*PI;
            }
            *(ori + k) = ori_tmp;   
            k++;
        }
    }
    return k;
}





/*
 * Interpolates the bin position given three values.  Returns the offset
 * bin position (assumes equally spaced bins)
 */
int
interpolate_parabola(double *offset, double y1, double y2, double y3)
{
    *offset =  0.5 * (y1 - y3) / (y1 - 2.0 * y2 + y3);
    
    return 0;
}





/*
 * For given keypoint location and orientation, returns SIFT descriptor
 */
int
keypoint_descriptor(double *D, double *gori, double *gmag,
                                double ori, double indexscale)
{
    int i, j, k;
   
    double x, y, X, Y, indX, indY;
    double distsq, weight, sigma_weight;
    double ori_tmp;
    double mag_tmp;
    double D_vect[128];
    double *D_vect_ptr;
    double sum_vect;
    double sum_vect2;
    int dindex, jx, iy;
    double first, middle, last;
    double Cori, Sori;      /* Sine and cosine of keypoint orientation */
    
    /* Interpolation for the index bins */
    int ind_bin[2][2];
    double weight_bin[2][2];
    double bindiff;
  
    /* Interpolation for the orientation histogram */
    double histbin;
    int ind_ori[2];
    double weight_ori[2];
    double histdiff;
    int Pw = (int)(PWIDTH);
    
    
    D_vect_ptr = D_vect; 
    
    /* ensure all values in the descriptor are set to zero */
    for (i = 0; i < 128; i++) {
        *(D_vect_ptr + i) = 0.0;
    }
    
    /* Set the Gaussian scaling for weighting values */
    sigma_weight = 0.5 * 4;     /* 4 = number of indexbins */
    
    /* Precompute the sine and cosine of the angles */
    Sori = sin(ori);
    Cori = cos(ori);
    
    for (i = -Pw; i <= Pw; i++) {
        y = (double)(i);
        
        for (j = -Pw; j <= Pw; j++, gori++, gmag++) {
            x = (double)(j);
            
            /* 
             * Get the indexbin position accounting for the rotation 
             * [X;Y] = R'[X;Y]
             */
            X = (x*Cori + y*Sori) / indexscale;
            Y = (-x*Sori + y*Cori) / indexscale;
            distsq = (X*X + Y*Y);       /* Index bin distance */
            indX = X + (indexbins/2.0) - 0.5;
            indY = Y + (indexbins/2.0) - 0.5;
            weight = exp(-distsq / (2.0 * sigma_weight*sigma_weight));
            
            /* Proceed only if pixel will contribute to the descriptor */
            if ((indX > -1) && (indX < indexbins) &&
                (indY > -1) && (indY < indexbins)) {
                    
                /* Set the integer values for the index bins */
                ind_bin[0][0] = (int)(floor(indX));
                ind_bin[0][1] = ind_bin[0][0] + 1;
                ind_bin[1][0] = (int)(floor(indY));
                ind_bin[1][1] = ind_bin[1][0] + 1;
           
                /* Find the weights for the interpolation of the indexbins */
                bindiff = indX - (double)(ind_bin[0][0]);
                weight_bin[0][0] = 1.0 - bindiff;
                weight_bin[0][1] = bindiff;
                bindiff = indY - (double)(ind_bin[1][0]);
                weight_bin[1][0] = 1.0 - bindiff;
                weight_bin[1][1] = bindiff;
            
                
                /* Get the gradient orientation, correct for keypoint
                 * orientation and ensure in the range 0 to 2pi
                 */
                ori_tmp = *gori - ori;
                while (ori_tmp < 0.0) {
                    ori_tmp += 2.0*PI;
                }
                while (ori_tmp > 2.0*PI) {
                    ori_tmp -= 2.0*PI;
                }
               
                /* Set the integer values for the orientation bins */
                histbin = 8.0 * (ori_tmp + 0.0001) / (2.0*PI);
                ind_ori[0] = (int)(floor(histbin));
                ind_ori[1] = ind_ori[0] + 1;
                
                /* Now set the weights for the interpolation */
                histdiff = histbin - (double)(ind_ori[0]);
                weight_ori[0] = 1.0 - histdiff;
                weight_ori[1] = histdiff;
                
                /* Do histogram bin wrap around if required */
                if (ind_ori[0] > 7) {
                    ind_ori[0] -= 8;
                }
                if (ind_ori[1] > 7) {
                    ind_ori[1] -= 8;
                }
                
                
                /* 
                 * Get the weighted gradient magnitude depending on the 
                 * distance from the keypoint 
                 */
                mag_tmp = *gmag * weight;                  
                
                /* Add the result for this pixel to the descriptor */
                add_to_descriptor(D_vect_ptr, ind_bin, weight_bin, ind_ori, 
                                                    weight_ori, mag_tmp);
            }
        }
    }
    
    /* Get Euclidean length of descriptor */
    sum_vect = 0.0;
    for (i = 0; i < 128; i++) {
        sum_vect += (D_vect[i] * D_vect[i]);
    }
    sum_vect = sqrt(sum_vect);
    
    for (i = 0; i < 128; i++) {
        D_vect[i] /= sum_vect;
    } 
    
    /* Add to the output descriptor (normalised) */
    for (i = 0; i < 128; i++) {
        *(D + i) = 512.0 * D_vect[i];
    }
                
    return 0;
}
    
    
    



/*
 * Given the index bins, histogram bins and weights, adds the given 
 * weighted gradient magnitude to the SIFT descriptor
 */
int 
add_to_descriptor(double *D, int ind_bin[2][2], 
                  double weight_bin[2][2], int ind_ori[2], 
                  double weight_ori[2], double mag)
{
    int i, j, k;
    int ind_x, ind_y, ind_h, ind_D;
    double weight_indexbin;
    double weight_histbin;
    double mag_tmp;
    
    for (i = 0; i < 2; i++) {
        ind_y = ind_bin[1][i];
      
        if ((ind_y >= 0) && (ind_y < indexbins)) {
        
            for (j = 0; j < 2; j++) {
                ind_x = ind_bin[0][j];
            
                if ((ind_x >= 0) && (ind_x < indexbins)) {
                    weight_indexbin = weight_bin[0][j] * weight_bin[1][i];
                    mag_tmp = mag * weight_indexbin;    
                
                    for (k = 0; k < 2; k++) {
                        ind_h = ind_ori[k];
                        ind_D = ind_h + ind_x*8 + ind_y*4*8;
                        *(D + ind_D) += mag_tmp * weight_ori[k];
                    }
                }
            }
        }
    }
    return 0;
}







/*
 * Applies a small 3x3 Gaussian kernel to the patch to try and reduce 
 * artifacts which occurred during the resampling process
 *=======================================================================*/
int
smooth_patch(double *patch, double *patch_tmp)
{
    int i, j;
    double kernel[5];
    double *input, *output;
    int Pw = (int)(PWIDTH);
    int N = 2*Pw+1;
    int r, c;
    
    kernel[0] = 0.0545;
    kernel[1] = 0.2442;
    kernel[2] = 0.4026;
    kernel[3] = 0.2442;
    kernel[4] = 0.0545;
    
    /*kernel[0] = 0.3554;
    kernel[1] = 0.8645;
    kernel[2] = 0.3554;
    */
    
    /* Do horizonal convolution */
    input = &patch[0];
    output = &patch_tmp[0];
    for (r = 0; r < N; r++) {
        for (i = 0; i < N; i++, input++, output++) {        
            if (i == 0) {
                *output = *input*kernel[0] + *input*kernel[1] + *input*kernel[2] +
                          *(input + 1)*kernel[3] + *(input + 2)*kernel[4];
            } else if (i == 1) {
                *output = *(input - 1)*kernel[0] + *(input - 1)*kernel[1] + 
                          *input*kernel[2] + *(input + 1)*kernel[3] +  
                          *(input + 2)*kernel[4];
            } else if (i == N-2) {
                *output = *(input-2)*kernel[0] + *(input - 1)*kernel[1] + 
                          *input*kernel[3] + *(input + 1)*kernel[3] + 
                          *(input + 1)*kernel[4];             
            } else if (i == N-1) {
                *output = *(input-2)*kernel[0] + *(input - 1)*kernel[1] + 
                          *input*kernel[2] + *input*kernel[3] + *input*kernel[4];
            } else {
                *output = *(input-2)*kernel[0] + *(input - 1)*kernel[1] + 
                          *input*kernel[2] + *(input + 1)*kernel[3] + 
                          *(input + 2)*kernel[4];
            }
        }
    }
    
    
    
    /* Do vertical convolution */
    for (c = 0; c < N; c++) {
        input = &patch_tmp[c];
        output = &patch[c];
        for (i = 0; i < N; i++, input+=N, output+=N) {        
            if (i == 0) {
                *output = *input*kernel[0] + *input*kernel[1] + *input*kernel[2] +
                          *(input + N)*kernel[3] + *(input + 2*N)*kernel[4];
            } else if (i == 1) {
                *output = *(input - N)*kernel[0] + *(input - N)*kernel[1] + 
                          *input*kernel[2] + *(input + N)*kernel[3] +  
                          *(input + 2*N)*kernel[4];
            } else if (i == N-2) {
                *output = *(input - 2*N)*kernel[0] + *(input - N)*kernel[1] + 
                          *input*kernel[3] + *(input + N)*kernel[3] + 
                          *(input + N)*kernel[4];             
            } else if (i == N-1) {
                *output = *(input-2*N)*kernel[0] + *(input - N)*kernel[1] + 
                          *input*kernel[2] + *input*kernel[3] + *input*kernel[4];
            } else {
                *output = *(input - 2*N)*kernel[0] + *(input - N)*kernel[1] + 
                          *input*kernel[2] + *(input + N)*kernel[3] + 
                          *(input + 2*N)*kernel[4];
            }
        }
    }
    return 0;
}







/* 
 * Test to see if the point is on an edge
 * Remember that Hessian is symmetric (hxy = hyx)
 */
int
not_edge_patch(double *patch) 
{
    double H_11, H_12, H_22;
    double detH, trH;
    double val1, val2;
    int Pw = (int)(PWIDTH);
    
    double D[3][3];
    int width = 3;
    int nr = 2*Pw + 1;
    int i, j;
    int ind_i, ind_j;
    
    double EDGE_RATIO = 10.0;
    
    /* Get the 3x3 region */
    for (i = 0; i <= 2; i++) {
        ind_i = Pw + (i-1)*width;
        for (j = 0; j <= 2; j++) {
            ind_j = Pw + (j-1)*width;       
            D[i][j] = *(patch + ind_i*nr + ind_j);
        }
    }
    
    /* Get the Hessian matrix */
    H_22 = D[1][2] + D[1][0] - 2.0 * D[1][1];
    H_11 = D[2][1] + D[0][1] - 2.0 * D[1][1];
    H_12 = ((D[2][2] - D[2][0]) - (D[0][2] - D[0][0])) / 4.0;
    
    /* Find the determinant and trace */
    detH = (H_11 * H_22) - (H_12 * H_12);
    trH = H_11 + H_22;
        
    /* See if the extrema is an edge response */  
    val1 = detH * (EDGE_RATIO + 1.0) * (EDGE_RATIO + 1.0);
    val2 = EDGE_RATIO * trH * trH;
    if (val1 > val2) {
        return 1;
    } else {
        return 0;
    }
}









/************************************************************************/
/*                                                                      */
/*  SPHERICAL (PARABOLIC) DIFFUSION KERNELS                             */
/*                                                                      */ 
/*  Some old hard coded kernels that were used in a lot of testing:     */
/*  Used Matlab to find values and print code                           */
/*                                                                      */
/*  Update code to take kernels as input                                */
/*                                                                      */
/************************************************************************/
/************************************************************************/


/*
 * Get the pre-calculated spherical diffusion kernels for all octaves and
 * scales - PARABOLIC SCALES
 *--------------------------------------------------------------------------*/
int
diffusion_kernel_para_scales(double *kernel, int octave, int scale)
{
    int i;
    double gsum;
    int width;
    double g[30];
                
    switch (octave) {
        case 1:
            switch (scale) {
                /* Presmooth Kernel */
                case 1:
                    width = 5;
                    g[0] = 0.31941094;
                    g[1] = 0.23182094;
                    g[2] = 0.08862646;
                    g[3] = 0.01784794;
                    g[4] = 0.00189338;
                    g[5] = 0.00010581;
                    break;
                case 2:
                    width = 5;
                    g[0] = 0.32532962;
                    g[1] = 0.23330253;
                    g[2] = 0.08604174;
                    g[3] = 0.01631921;
                    g[4] = 0.00159184;
                    g[5] = 0.00007986;
                    break;
                case 3:
                    width = 7;
                    g[0] = 0.25821285;
                    g[1] = 0.20941560;
                    g[2] = 0.11171306;
                    g[3] = 0.03919825;
                    g[4] = 0.00904699;
                    g[5] = 0.00137349;
                    g[6] = 0.00013717;
                    g[7] = 0.00000901;
                    break;
                case 4:
                    width = 8;
                    g[0] = 0.20494488;
                    g[1] = 0.17960996;
                    g[2] = 0.12089582;
                    g[3] = 0.06250032;
                    g[4] = 0.02481691;
                    g[5] = 0.00756856;
                    g[6] = 0.00177291;
                    g[7] = 0.00031899;
                    g[8] = 0.00004409;
                    break;
                case 5:
                    width = 10;
                    g[0] = 0.16266495;
                    g[1] = 0.14969013;
                    g[2] = 0.11665173;
                    g[3] = 0.07698210;
                    g[4] = 0.04302208;
                    g[5] = 0.02036105;
                    g[6] = 0.00816057;
                    g[7] = 0.00276987;
                    g[8] = 0.00079620;
                    g[9] = 0.00019383;
                    g[10] = 0.00003996;
                    break;
                case 6:
                    width = 13;
                    g[0] = 0.12910550;
                    g[1] = 0.12251880;
                    g[2] = 0.10470733;
                    g[3] = 0.08058770;
                    g[4] = 0.05585725;
                    g[5] = 0.03486677;
                    g[6] = 0.01960056;
                    g[7] = 0.00992324;
                    g[8] = 0.00452450;
                    g[9] = 0.00185792;
                    g[10] = 0.00068711;
                    g[11] = 0.00022886;
                    g[12] = 0.00006866;
                    g[13] = 0.00001855;
                    break;
            }
            break;
        case 2:
            switch (scale) {
                /* Presmooth Kernel */
                case 1:
                    width = 5;
                    g[0] = 0.31940684;
                    g[1] = 0.23181885;
                    g[2] = 0.08862795;
                    g[3] = 0.01784987;
                    g[4] = 0.00189401;
                    g[5] = 0.00010589;
                    break;
                case 2:
                    width = 5;
                    g[0] = 0.32532596;
                    g[1] = 0.23330064;
                    g[2] = 0.08604313;
                    g[3] = 0.01632095;
                    g[4] = 0.00159238;
                    g[5] = 0.00007992;
                    break;
                case 3:
                    width = 7;
                    g[0] = 0.25820824;
                    g[1] = 0.20941242;
                    g[2] = 0.11171331;
                    g[3] = 0.03920070;
                    g[4] = 0.00904891;
                    g[5] = 0.00137420;
                    g[6] = 0.00013731;
                    g[7] = 0.00000903;
                    break;
                case 4:
                    width = 8;
                    g[0] = 0.20493908;
                    g[1] = 0.17960530;
                    g[2] = 0.12089426;
                    g[3] = 0.06250210;
                    g[4] = 0.02482008;
                    g[5] = 0.00757103;
                    g[6] = 0.00177411;
                    g[7] = 0.00031939;
                    g[8] = 0.00004418;
                    break;
                case 5:
                    width = 10;
                    g[0] = 0.16265766;
                    g[1] = 0.14968374;
                    g[2] = 0.11664795;
                    g[3] = 0.07698187;
                    g[4] = 0.04302486;
                    g[5] = 0.02036504;
                    g[6] = 0.00816404;
                    g[7] = 0.00277206;
                    g[8] = 0.00079727;
                    g[9] = 0.00019424;
                    g[10] = 0.00004009;
                    break;
                case 6:
                    width = 13;
                    g[0] = 0.12909630;
                    g[1] = 0.12251032;
                    g[2] = 0.10470098;
                    g[3] = 0.08058459;
                    g[4] = 0.05585774;
                    g[5] = 0.03487018;
                    g[6] = 0.01960545;
                    g[7] = 0.00992810;
                    g[8] = 0.00452835;
                    g[9] = 0.00186045;
                    g[10] = 0.00068853;
                    g[11] = 0.00022955;
                    g[12] = 0.00006895;
                    g[13] = 0.00001866;
                    break;
            }
            break;
        case 3:
            switch (scale) {
                case 2:
                    width = 5;
                    g[0] = 0.32531135;
                    g[1] = 0.23329306;
                    g[2] = 0.08604867;
                    g[3] = 0.01632788;
                    g[4] = 0.00159452;
                    g[5] = 0.00008018;
                    break;
                case 3:
                    width = 7;
                    g[0] = 0.25818982;
                    g[1] = 0.20939970;
                    g[2] = 0.11171427;
                    g[3] = 0.03921048;
                    g[4] = 0.00905660;
                    g[5] = 0.00137705;
                    g[6] = 0.00013790;
                    g[7] = 0.00000910;
                    break;
                case 4:
                    width = 8;
                    g[0] = 0.20491590;
                    g[1] = 0.17958668;
                    g[2] = 0.12088801;
                    g[3] = 0.06250922;
                    g[4] = 0.02483276;
                    g[5] = 0.00758092;
                    g[6] = 0.00177893;
                    g[7] = 0.00032098;
                    g[8] = 0.00004455;
                    break;
                case 5:
                    width = 10;
                    g[0] = 0.16262846;
                    g[1] = 0.14965817;
                    g[2] = 0.11663282;
                    g[3] = 0.07698097;
                    g[4] = 0.04303596;
                    g[5] = 0.02038099;
                    g[6] = 0.00817791;
                    g[7] = 0.00278085;
                    g[8] = 0.00080157;
                    g[9] = 0.00019591;
                    g[10] = 0.00004062;
                    break;
                case 6:
                    width = 13;
                    g[0] = 0.12905949;
                    g[1] = 0.12247640;
                    g[2] = 0.10467556;
                    g[3] = 0.08057214;
                    g[4] = 0.05585967;
                    g[5] = 0.03488379;
                    g[6] = 0.01962500;
                    g[7] = 0.00994755;
                    g[8] = 0.00454374;
                    g[9] = 0.00187060;
                    g[10] = 0.00069425;
                    g[11] = 0.00023233;
                    g[12] = 0.00007013;
                    g[13] = 0.00001910;
                    break;
            }
            break;
        case 4:
            switch (scale) {
                case 2:
                    width = 5;
                    g[0] = 0.32525288;
                    g[1] = 0.23326274;
                    g[2] = 0.08607085;
                    g[3] = 0.01635562;
                    g[4] = 0.00160312;
                    g[5] = 0.00008123;
                    break;
                case 3:
                    width = 7;
                    g[0] = 0.25811611;
                    g[1] = 0.20934880;
                    g[2] = 0.11171809;
                    g[3] = 0.03924957;
                    g[4] = 0.00908737;
                    g[5] = 0.00138848;
                    g[6] = 0.00014025;
                    g[7] = 0.00000939;
                    break;
                case 4:
                    width = 8;
                    g[0] = 0.20482310;
                    g[1] = 0.17951214;
                    g[2] = 0.12086296;
                    g[3] = 0.06253762;
                    g[4] = 0.02488349;
                    g[5] = 0.00762052;
                    g[6] = 0.00179825;
                    g[7] = 0.00032742;
                    g[8] = 0.00004607;
                    break;
                case 5:
                    width = 10;
                    g[0] = 0.16251157;
                    g[1] = 0.14955582;
                    g[2] = 0.11657219;
                    g[3] = 0.07697728;
                    g[4] = 0.04308025;
                    g[5] = 0.02044476;
                    g[6] = 0.00823345;
                    g[7] = 0.00281612;
                    g[8] = 0.00081890;
                    g[9] = 0.00020269;
                    g[10] = 0.00004276;
                    break;
                case 6:
                    width = 13;
                    g[0] = 0.12891207;
                    g[1] = 0.12234054;
                    g[2] = 0.10457373;
                    g[3] = 0.08052214;
                    g[4] = 0.05586720;
                    g[5] = 0.03493804;
                    g[6] = 0.01970308;
                    g[7] = 0.01002535;
                    g[8] = 0.00460545;
                    g[9] = 0.00191147;
                    g[10] = 0.00071738;
                    g[11] = 0.00024367;
                    g[12] = 0.00007498;
                    g[13] = 0.00002093;
                    break;
            }
            break;
        case 5:
            switch (scale) {
                case 2:
                    width = 5;
                    g[0] = 0.32501882;
                    g[1] = 0.23314127;
                    g[2] = 0.08615935;
                    g[3] = 0.01646668;
                    g[4] = 0.00163778;
                    g[5] = 0.00008551;
                    break;
                case 3:
                    width = 7;
                    g[0] = 0.25782088;
                    g[1] = 0.20914482;
                    g[2] = 0.11173298;
                    g[3] = 0.03940567;
                    g[4] = 0.00921077;
                    g[5] = 0.00143473;
                    g[6] = 0.00014997;
                    g[7] = 0.00001061;
                    break;
                case 4:
                    width = 8;
                    g[0] = 0.20445125;
                    g[1] = 0.17921331;
                    g[2] = 0.12076203;
                    g[3] = 0.06265038;
                    g[4] = 0.02508600;
                    g[5] = 0.00777949;
                    g[6] = 0.00187663;
                    g[7] = 0.00035399;
                    g[8] = 0.00005253;
                    break;
                case 5:
                    width = 10;
                    g[0] = 0.16204270;
                    g[1] = 0.14914512;
                    g[2] = 0.11632834;
                    g[3] = 0.07696093;
                    g[4] = 0.04325579;
                    g[5] = 0.02069903;
                    g[6] = 0.00845633;
                    g[7] = 0.00295920;
                    g[8] = 0.00089040;
                    g[9] = 0.00023135;
                    g[10] = 0.00005216;
                    break;
                case 6:
                    width = 13;
                    g[0] = 0.12831964;
                    g[1] = 0.12179440;
                    g[2] = 0.10416375;
                    g[3] = 0.08031930;
                    g[4] = 0.05589407;
                    g[5] = 0.03515178;
                    g[6] = 0.02001321;
                    g[7] = 0.01033656;
                    g[8] = 0.00485478;
                    g[9] = 0.00207907;
                    g[10] = 0.00081428;
                    g[11] = 0.00029260;
                    g[12] = 0.00009680;
                    g[13] = 0.00002959;
                    break;
            }
            break;
    }

     
    /* 
     * Copy contents over to kernel and ensure unit volume 
     * (only need half the kernel including middle value)
     */
    gsum = 0.0;
    for (i = -width; i <= width; i++) {
        gsum += g[(int)(fabs(i))];
    }
    for (i = 0; i <= width; i++) {
        *(kernel + i) = g[i] / gsum;
    }
    return width;
}








/*
 * Get the pre-calculated spherical diffusion kernels for all octaves and
 * scales - FISHEYE SCALES
 *--------------------------------------------------------------------------*/
int
diffusion_kernel_fish_scales(double *kernel, int octave, int scale)
{
    int i;
    double gsum;
    int width;
    double g[30];
                
    switch (octave) {
        case 1:
            switch (scale) {
                /* Presmooth Kernel */
                case 1:
                    width = 4;
                    g[0] = 0.46616045;
                    g[1] = 0.23553060;
                    g[2] = 0.03037944;
                    g[3] = 0.00100078;
                    g[4] = 0.00000895;
                    break;
                case 2:
                    width = 4;
                    g[0] = 0.47479783;
                    g[1] = 0.23384654;
                    g[2] = 0.02793771;
                    g[3] = 0.00081033;
                    g[4] = 0.00000650;
                    break;
                case 3:
                    width = 5;
                    g[0] = 0.37685139;
                    g[1] = 0.24121538;
                    g[2] = 0.06325756;
                    g[3] = 0.00679675;
                    g[4] = 0.00029922;
                    g[5] = 0.00000540;
                    break;
                case 4:
                    width = 6;
                    g[0] = 0.29910683;
                    g[1] = 0.22581981;
                    g[2] = 0.09717868;
                    g[3] = 0.02383739;
                    g[4] = 0.00333299;
                    g[5] = 0.00026565;
                    g[6] = 0.00001207;
                    break;
                case 5:
                    width = 7;
                    g[0] = 0.23740190;
                    g[1] = 0.19887885;
                    g[2] = 0.11692382;
                    g[3] = 0.04824261;
                    g[4] = 0.01396940;
                    g[5] = 0.00283891;
                    g[6] = 0.00040491;
                    g[7] = 0.00004053;
                    break;
                case 6:
                    width = 9;
                    g[0] = 0.18842523;
                    g[1] = 0.16853805;
                    g[2] = 0.12060781;
                    g[3] = 0.06905159;
                    g[4] = 0.03162978;
                    g[5] = 0.01159173;
                    g[6] = 0.00339888;
                    g[7] = 0.00079738;
                    g[8] = 0.00014968;
                    g[9] = 0.00002248;
                    break;
            }
            break;
        case 2:
            switch (scale) {
                /* Presmooth Kernel */
                case 1:
                    width = 4;
                    g[0] = 0.46616119;
                    g[1] = 0.23553000;
                    g[2] = 0.03038046;
                    g[3] = 0.00100054;
                    g[4] = 0.00000841;
                    break;
                case 2:
                    width = 4;
                    g[0] = 0.47480012;
                    g[1] = 0.23384583;
                    g[2] = 0.02793859;
                    g[3] = 0.00080982;
                    g[4] = 0.00000570;
                    break;
                case 3:
                    width = 5;
                    g[0] = 0.37684824;
                    g[1] = 0.24121421;
                    g[2] = 0.06325919;
                    g[3] = 0.00679772;
                    g[4] = 0.00029935;
                    g[5] = 0.00000540;
                    break;
                case 4:
                    width = 6;
                    g[0] = 0.29910285;
                    g[1] = 0.22581747;
                    g[2] = 0.09717975;
                    g[3] = 0.02383950;
                    g[4] = 0.00333394;
                    g[5] = 0.00026583;
                    g[6] = 0.00001209;
                    break;
                case 5:
                    width = 7;
                    g[0] = 0.23739689;
                    g[1] = 0.19887516;
                    g[2] = 0.11692347;
                    g[3] = 0.04824499;
                    g[4] = 0.01397189;
                    g[5] = 0.00284016;
                    g[6] = 0.00040528;
                    g[7] = 0.00004060;
                    break;
                case 6:
                    width = 9;
                    g[0] = 0.18841892;
                    g[1] = 0.16853279;
                    g[2] = 0.12060548;
                    g[3] = 0.06905277;
                    g[4] = 0.03163304;
                    g[5] = 0.01159490;
                    g[6] = 0.00340083;
                    g[7] = 0.00079823;
                    g[8] = 0.00014994;
                    g[9] = 0.00002254;
                    break;
            }
            break;
        case 3:
            switch (scale) {
                case 2:
                    width = 4;
                    g[0] = 0.47479010;
                    g[1] = 0.23384512;
                    g[2] = 0.02794357;
                    g[3] = 0.00081056;
                    g[4] = 0.00000571;
                    break;
                case 3:
                    width = 5;
                    g[0] = 0.37683561;
                    g[1] = 0.24120954;
                    g[2] = 0.06326571;
                    g[3] = 0.00680163;
                    g[4] = 0.00029989;
                    g[5] = 0.00000543;
                    break;
                case 4:
                    width = 6;
                    g[0] = 0.29908695;
                    g[1] = 0.22580809;
                    g[2] = 0.09718401;
                    g[3] = 0.02384795;
                    g[4] = 0.00333775;
                    g[5] = 0.00026657;
                    g[6] = 0.00001216;
                    break;
                case 5:
                    width = 7;
                    g[0] = 0.23737687;
                    g[1] = 0.19886040;
                    g[2] = 0.11692205;
                    g[3] = 0.04825451;
                    g[4] = 0.01398185;
                    g[5] = 0.00284515;
                    g[6] = 0.00040675;
                    g[7] = 0.00004087;
                    break;
                case 6:
                    width = 9;
                    g[0] = 0.18839369;
                    g[1] = 0.16851177;
                    g[2] = 0.12059615;
                    g[3] = 0.06905748;
                    g[4] = 0.03164609;
                    g[5] = 0.01160759;
                    g[6] = 0.00340864;
                    g[7] = 0.00080161;
                    g[8] = 0.00015102;
                    g[9] = 0.00002280;
                    break;
            }
            break;
        case 4:
            switch (scale) {
                case 2:
                    width = 4;
                    g[0] = 0.47475000;
                    g[1] = 0.23384226;
                    g[2] = 0.02796347;
                    g[3] = 0.00081349;
                    g[4] = 0.00000578;
                    break;
                case 3:
                    width = 5;
                    g[0] = 0.37678511;
                    g[1] = 0.24119085;
                    g[2] = 0.06329179;
                    g[3] = 0.00681724;
                    g[4] = 0.00030204;
                    g[5] = 0.00000552;
                    break;
                case 4:
                    width = 6;
                    g[0] = 0.29902331;
                    g[1] = 0.22577059;
                    g[2] = 0.09720107;
                    g[3] = 0.02388173;
                    g[4] = 0.00335300;
                    g[5] = 0.00026952;
                    g[6] = 0.00001243;
                    break;
                case 5:
                    width = 7;
                    g[0] = 0.23729674;
                    g[1] = 0.19880132;
                    g[2] = 0.11691634;
                    g[3] = 0.04829253;
                    g[4] = 0.01402168;
                    g[5] = 0.00286515;
                    g[6] = 0.00041265;
                    g[7] = 0.00004196;
                    break;
                case 6:
                    width = 9;
                    g[0] = 0.18829270;
                    g[1] = 0.16842760;
                    g[2] = 0.12055877;
                    g[3] = 0.06907629;
                    g[4] = 0.03169820;
                    g[5] = 0.01165837;
                    g[6] = 0.00343995;
                    g[7] = 0.00081523;
                    g[8] = 0.00015539;
                    g[9] = 0.00002386;
                    break;
            }
            break;
        case 5:
            switch (scale) {
                case 2:
                    width = 4;
                    g[0] = 0.47458957;
                    g[1] = 0.23383076;
                    g[2] = 0.02804313;
                    g[3] = 0.00082528;
                    g[4] = 0.00000604;
                    break;
                case 3:
                    width = 5;
                    g[0] = 0.37658298;
                    g[1] = 0.24111599;
                    g[2] = 0.06339599;
                    g[3] = 0.00687985;
                    g[4] = 0.00031077;
                    g[5] = 0.00000591;
                    break;
                case 4:
                    width = 6;
                    g[0] = 0.29876855;
                    g[1] = 0.22562037;
                    g[2] = 0.09726901;
                    g[3] = 0.02401686;
                    g[4] = 0.00341435;
                    g[5] = 0.00028154;
                    g[6] = 0.00001359;
                    break;
                case 5:
                    width = 7;
                    g[0] = 0.23697577;
                    g[1] = 0.19856456;
                    g[2] = 0.11689301;
                    g[3] = 0.04844417;
                    g[4] = 0.01418117;
                    g[5] = 0.00294584;
                    g[6] = 0.00043680;
                    g[7] = 0.00004656;
                    break;
                case 6:
                    width = 9;
                    g[0] = 0.18788786;
                    g[1] = 0.16809004;
                    g[2] = 0.12040834;
                    g[3] = 0.06915041;
                    g[4] = 0.03190586;
                    g[5] = 0.01186176;
                    g[6] = 0.00356644;
                    g[7] = 0.00087106;
                    g[8] = 0.00017371;
                    g[9] = 0.00002845;
                    break;
            }
            break;
    }

     
    /* 
     * Copy contents over to kernel and ensure unit volume 
     * (only need half the kernel including middle value)
     */
    gsum = 0.0;
    for (i = -width; i <= width; i++) {
        gsum += g[(int)(fabs(i))];
    }
    for (i = 0; i <= width; i++) {
        *(kernel + i) = g[i] / gsum;
    }
    return width;
}





