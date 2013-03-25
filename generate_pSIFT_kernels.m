% 
%   pSIFT_vars = generate_pSIFT_kernels(pSIFT_vars)
%
%   Computes the pSIFT kernels and saves
%
%   Input pSIFT_vars must have fields:
%       m = distance stereographic plane from center view sphere
%       noct = the number of octaves of scale-space
%       spo = scale per octave (3 is a good value)
%
%   Finds the discrete diffusion kernel values
%   The kernels are the stereographic projection of the spherical
%   Gaussian to the image plane.
%
%   Ouput pSIFT_vars has following fields added:
%       kernels = matrix of all 1D kernels required
%       kernels_widths = width of each kernel
%       kt_scales = the (spherical) scale of each kernel       
%       m_octave = the distance of stereographic image from center
%                  view sphere for each octave
%
%   NOTE: The kernel values are computed with a summation up to l=2048
%         (this exceeds the maximum bandwidth of most images)
%   
%   The initial starting scale is found automatically based on 
%   the initial scale/presmooth scale used by SIFT
%
%   Copyright (C) 2013 Peter Hansen [phansen.au(at)gmail.com]  
%


function pSIFT_vars = generate_pSIFT_kernels(pSIFT_vars)
    
    % Spherical harmonic bandwidth
    b = 2048;       % Should be larger than image bandwidth
    
    noct = pSIFT_vars.noct;
    spo = pSIFT_vars.spo;
    fprintf('Computing convolution kernels: %d octaves, %d scales per octave:\n', noct, spo);
    
    % For each octave, find the stereographic parameter m
    %----------------------------------------------------------------
    m_octave(2) = pSIFT_vars.m;     % 2nd octave is original image size
    m_octave(1) = 2*m_octave(2) + 1;
    for oct = 3:(noct+1)
        m_octave(oct) = (1/2)*(m_octave(oct-1) + 1) - 1;
    end
    
    
    % Get the pre-smooth scale, and all remaining scale.
    % kt values are in cell array {octave}{scale}
    % Scale found using double image option
    %----------------------------------------------------------------
    [kt_presmooth,kt] = set_pSIFT_start_scales(pSIFT_vars.m, ...
                                               pSIFT_vars.noct, pSIFT_vars.spo);
  
    
    % The presmooth kernel
    %----------------------------------------------------------------
    fprintf('\tKernel: presmooth\n');
    kernel_presmooth = find_kernel_values(kt_presmooth, m_octave(1), b);
    kernel_presmooth_width = length(kernel_presmooth);
    
    
    % The remaining scale-space kernels
    % * Semi-group is used in pSIFT, so find _difference_ of scale-space
    %----------------------------------------------------------------
    nmax = kernel_presmooth_width;
    for oct = 1:noct
        kernel{oct}{1} = [];
        kernel_width{oct}{1} = 0;
        for i = 2:(spo+3)
            fprintf('\tKernel: octave = %d  scale index %d\n', oct, i);
            kt_increase = kt{oct}{i} - kt{oct}{i-1};
            kernel{oct}{i} = find_kernel_values(kt_increase, m_octave(oct), b);
            ntmp = length(kernel{oct}{i});
            kernel_width{oct}{i} = ntmp;
            if ntmp > nmax; nmax = ntmp; end;
        end
    end
    
    
    % Put all the kernels into a single mxn array, and put all scales
    % into an array (row = octave, col = scale)
    %----------------------------------------------------------------
    kernels = zeros(noct * (spo+3) + 1, nmax);
    kernels_widths = zeros(noct * (spo+3) + 1, 1);
    kt_scales = zeros(noct, spo+3);
    
    kernels(1,1:kernel_presmooth_width) = kernel_presmooth;
    kernels_widths(1) = kernel_presmooth_width;
    n = 1;
    for oct = 1:noct
        for i = 1:(spo+3)
            n = n + 1;
            if i > 1
                kernels(n, 1:kernel_width{oct}{i}) = kernel{oct}{i};
                kernels_widths(n) = kernel_width{oct}{i};
            end
            kt_scales(oct, i) = kt{oct}{i};
        end
    end
    
    % Add all required data to pSIFT_vars
    pSIFT_vars.kernels = kernels;
    pSIFT_vars.kernels_widths = kernels_widths;
    pSIFT_vars.kt_scales = kt_scales;
    pSIFT_vars.m_octave = m_octave;
    
end
    


% Finds the spherical diffusion kernel mapped, via stereographic 
% projection, to the image (up to 4 sigma)
%
% This function will take some time...
%
%====================================================================
function kernel = find_kernel_values(kt, mp, b)

    % Find the radius on the image plane to include 4 sigma
    sigma = sqrt(2*kt);
    U = map_unified_sphere2img(sin(4*sigma), 0, cos(4*sigma), mp, 1);
    Rk = ceil(U(1));

    % Find the angles on the sphere for each of these pixels
    R = 0:Rk;
    theta = 2 * atan(R / (mp + 1));

    % Find the diffusion function for the given scale for all theta values
    Ctheta = cos(theta);
    kernel = zeros(1,length(theta));
    for l = 0:b
        sqrtval = sqrt((2*l+1)/(4*pi));  
        P = legendre(l,Ctheta);
        P_zonal = P(1,:);
        Y = P_zonal * sqrtval;
        vals = sqrtval * Y * exp(-l*(l+1)*kt);
        kernel = kernel + vals;
    end
    kernel_full = [fliplr(kernel(2:end)) kernel];
    kernel = kernel / sum(kernel_full);
end
%====================================================================




%
%   [kt_presmooth, kt] = set_pSIFT_start_scales(cam, noct, spo)
%
%   Given the camera model, selects the scales to use for pSIFT.
%   The initial scale is found by projecting the SIFT starting scale
%   on the wide-angle image plane to the sphere.  As diffusion is
%   implemented on the parabolic image plane, need to find the presmoothing
%   kernel scales.
%
%   A double image option is uesd:
%   Assumes initial scale 0.5 and starting scale 0.8 wrt. 
%   the original image size.
%
%   Returns the presmooth scale kt_presmooth and the scales
%   kt{octave}{scale}.  
%
function [kt_presmooth, kt] = set_pSIFT_start_scales(m_cam, noct, spo)

    % Set multiplication factor
    k = 2^(1/spo);

   
    % Find the scales kt
    %----------------------------------------------------------------
    % Get the starting scale
    S = map_unified_img2sphere(0.8, 0, m_cam, 1);
    sigma(1) = acos(S(3)) / sqrt(2); 
    
    
    % Find all other scales using the multiplicative factor
    for i = 2:(noct*spo + 3)
        sigma(i) = sigma(i-1)*k;
    end
    kt_tmp = sigma.^2;
    
    
    % Group into octaves - for each octave need spo+3 scales
    for i = 1:noct
        ind_start = (i-1)*spo;
        for j = 1:spo+3
            kt{i}{j} = kt_tmp(ind_start + j);
        end
    end
    
    
    % The presmooth scale
    %---------------------------------------------------------------------
    % Double image size
    S = map_unified_img2sphere(0.5, 0, m_cam, 1);
    sigma_init = acos(S(3)) / sqrt(2);
    kt_init = sigma_init^2;
    kt_presmooth = kt{1}{1} - kt_init;
    
end



% function S = map_unified_img2sphere(x, y, m, l)
%
% Maps point x,y in the image to a unit radius sphere.  The mapping is 
% given by the position of the image plane m and the point of projection l
%
% Returns the point on the sphere S
%
% This is a generic function for the unified image model
%
%====================================================================

function [varargout] = map_unified_img2sphere(x, y, m, l)

    if ((x==0) & (y==0))
        sx = 0;
        sy = 0;
        sz = 1;
    else
        theta = atan2(y,x);
        R = sqrt(x^2 + y^2);
              
        if l == 0                   % Perspective
            alpha = atan(R/m);
            r = sin(alpha);
            sx = r * cos(theta);
            sy = r * sin(theta);
            sz = cos(alpha);
        else 
            % Get equation to line
            %k = (m+l)/R;

            % Solve to find the intercept
            A = ((m+l)/R)^2 + 1;
            B = -2*l*((m+l)/R);
            C = l^2 - 1;
            r = (-B + sqrt(B^2 - 4*A*C)) / (2*A);

            % Find the unit cartesian coordinate on sphere
            sx = r * cos(theta);
            sy = r * sin(theta);
            sz = ((m+l)/R)*r - l;
        end
    end
    
    if nargout == 1
        varargout = {[sx,sy,sz]};
    else
        varargout(1) = {sx};
        varargout(2) = {sy};
        varargout(3) = {sz};
    end
end




% function P = map_sphere2img(x, y, z, F, L)
%
% Maps point x,y,z on a unit radius sphere to the image plane. The 
% mapping is  given by the focal length F and point of projection L 
% of the sphere
% 
% Returns the point on the image plane P
%
% Another generic function for unified image model
%
function [varargout] = map_unified_sphere2img(x, y, z, m, l)

    if (z-l) == 0
        scale_factor = 0;
    else
        scale_factor = (l+m) / (l+z);
    end
        
    u = x * scale_factor; 
    v = y * scale_factor;
    

    if nargout == 1
        varargout = {[u;v]};
    else
        varargout(1) = {u};
        varargout(2) = {v};
    end
end
