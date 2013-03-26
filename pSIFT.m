%
%   keypoints = pSIFT(img, mask, vars)
%
%   img = image obtained with stereographic projection
%   mask = image mask (0 invalid, 1 valid)
%   vars = structure of required variables   
%
%   vars needs fields:
%       pp: 2x1 vector, principal point of input image
%       m:  distance input stereographic image plane from center view sphere
%       noct: number of octaves 
%       spo: scales per octave
%       DoG_thresh: difference of Gaussian threshold (image vals 0-1),
%       recommended value = 0.0125
%       kernels: matrix of presmooth and convolution kernels
%       kernels_widths: vector of kernel support sizes
%       kt_scales: matrix of octave/spo diffusion scales
%       m_octave: distances of stereographic plane from center view
%                 sphere for each octave of scale space
%   
%
%   Copyright (C) 2013 Peter Hansen [phansen.au(at)gmail.com]  
%
%   For more details, see:
%   - "Wide-Angle Visual Feature Matching for Outdoor Localization"
%      Hansen, Corke and Boles
%      Int. Journal of Robotics Research, Vol.29, 2010.
%   http://ijr.sagepub.com/content/29/2-3/267.abstract
%

function f = pSIFT(img, mask, vars)
    

    % Ensure image is greyscale, and image and mask are same size
    [nr,nc,nd] = size(img);
    if nd ~= 1
        error('Image must be greyscale');
    end
    [nr_mask, nc_mask, nd_mask] = size(mask);
    if (nr_mask ~= nr) | (nc_mask ~= nc) | (nd_mask ~= 1)
        error('Mask is incorrect size');
    end
    
    % Call the pSIFT mex function.
    % Pre-transpose the image, kernels, and scales
    tstart = tic;
    fraw = pSIFT_mex(img'/ 255, mask', vars.kernels', vars.kernels_widths', ...
                        vars.kt_scales', vars.noct, vars.spo, vars.m_octave, ...
                        vars.DoG_thresh, vars.pp(1), vars.pp(2));
    dt = toc(tstart);
    nfeatures = size(fraw,2);
    fprintf('Found %d keypoints in %0.3f seconds\n', nfeatures, dt);
    
    
    % Get the spherical coords of the keypoints
    % Should really have the mex function output this directly
    N = size(fraw,2);
    x = (fraw(1:2,:) - repmat(vars.pp, [1 N])) / (vars.m + 1);
    sumxsq = sum(x.^2);
    den = 1 + sumxsq;
    S = [2*x(1,:)./den ; 2*x(2,:)./den; (1 - sumxsq)./den];
    
     
    % fraw is a matrix with all data
    % Put into a matlab struct, including spherical coord of point
    for n = 1:length(fraw(1,:))
        f(n).U = fraw(1:2,n);               % Matlab coords (not C)
        f(n).S = S(:,n);                    % Spherical
        f(n).kt = fraw(3,n)^2;              % Scale kt
        f(n).rot = fraw(4,n);               % Rotation
        f(n).desc = fraw(5:end,n);          % 128 dim SIFT descriptor
        f(n).ind = n;                       % Just an index
    end
     
end
   
