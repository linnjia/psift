%
%   pSIFT_demo
%
%   Simple demo to illustrate pSIFT
%
%   General process is:
%       1) Find mapping from original image to stereographic image
%       2) Setup pSIFT variables and get kernels
%       3) Create stereographic image
%       4) Run pSIFT
%
%   * Step 1 depends on the original camera model.  It's
%     necessary to back project the original image coordiantes to
%     the unit view sphere, and then project to a new image plane
%     using stereographic projection.  This new 'stereographic' image
%     is used by pSIFT.
%
%   * In this example, the original camera model used is the 
%     'unified' image model:
%      - "Catadioptric Projective Geometry"
%         Geyer and Daniilidis 
%         IJCV, 45(3):223--243, 2001.
%
%   For more details, see:
%   - "Wide-Angle Visual Feature Matching for Outdoor Localization"
%      Hansen, Corke and Boles
%      Int. Journal of Robotics Research, Vol.29, 2010.
%   http://ijr.sagepub.com/content/29/2-3/267.abstract
%   
%   NOTE: using matlab syle coords: upper left pixel position is (1,1)
%

clear all

fname_vars = 'pSIFT_vars.mat';
vars_loaded = false;
if exist(fname_vars, 'file')
    load(fname_vars);
    vars_loaded = true;
end


% Load sample image
img = double(imread('img_sample.png'));
[nr,nc,nd] = size(img);
if nd == 3; img = sum(img,3)/3; end


% 1) Find mapping from original image to stereographic image
%    The example here is specific to the fisheye camera model
%====================================================================
cam_fisheye.pp = [528.70; 384.50];  % Principal point   
cam_fisheye.m = 975;                % Distance of image plance from center view sphere
cam_fisheye.l = 2.7;                % Point of projection above center view sphere

[fisheye_u, fisheye_v] = meshgrid(1:nc, 1:nr);
[para_u, para_v, mask, mask_double, m_para] = fish2para_coords_mex(nr, nc, ...
            cam_fisheye.pp(1), cam_fisheye.pp(2), cam_fisheye.m, cam_fisheye.l);

% The stereographic image params
if ~vars_loaded
    pSIFT_vars.pp = cam_fisheye.pp;
    pSIFT_vars.m = m_para;
end

        
% 2) Setup (remaining) pSIFT variables and get kernels
%    Will take several minutes to find kernels
%====================================================================
if ~vars_loaded
    pSIFT_vars.noct = 5;                % Number of octaves (suggested default)
    pSIFT_vars.spo = 3;                 % Scales per octave (suggested default)
    pSIFT_vars.DoG_thresh = 0.0125;     % Diff Gaussian threshold (image data in range 0-1)
    pSIFT_vars = generate_pSIFT_kernels(pSIFT_vars);    % Kernel data
    save(fname_vars, 'pSIFT_vars');
end


% 3) Create stereographic image
%====================================================================
img_para = interp2(fisheye_u, fisheye_v, img, para_u, para_v, 'linear');

figure(1); cla; colormap(gray(255));
imagesc([img img_para img_para.*mask]); 
title('Original / Stereographic / Stereographic (masked)')
axis image; drawnow



% 4) Run pSIFT
%    There is a wrapper pSIFT.m for the mex code pSIFT_mex.c
%    Input image must be double with values in range 0-255
%    (pSIFT.m will rescale to 0-1)
%====================================================================
keypoints = pSIFT(img_para, mask, pSIFT_vars);

% Plot just keypoint locations
pSIFT_disp_keypoints(img_para.*mask, keypoints, 2); 

% Plot locations and orientation/scale vectors
pSIFT_disp_keypoints(img_para.*mask, keypoints, pSIFT_vars, 'vector', 3);

% Plot locations and support regions (circles on sphere)
pSIFT_disp_keypoints(img_para.*mask, keypoints, pSIFT_vars, 'support', 4);


