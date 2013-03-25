%
%   pSIFT_disp_keypoints(img, keypoints, fig)
%   pSIFT_disp_keypoints(img, keypoints, vars, keystyle, fig)
%
%   Plots the set of pSIFT keypoints in fig
%
%   pSIFT_disp_keypoints(img, keypoints, fig)
%   - displays just the locations of the keypoints
%
%   pSIFT_disp_keypoints(img, keypoints, vars, keystyle, fig)
%   - displays the locations of the keypoints and
%       keystyle = 'vector': scale and rotion as vector
%       keystyle = 'support': spherical circular support region as it
%                             appears in the image
%   
%   Copyright (C) 2013 Peter Hansen [phansen.au(at)gmail.com]
%

function pSIFT_disp_keypoints(img, keypoints, varargin)

    if nargin == 3
        fig = varargin{1};
    elseif nargin == 5
        vars = varargin{1};
        keystyle = varargin{2};
        fig = varargin{3};
    else
        error('Function takes 3 or 5 inputs')
    end
    
    
    figure(fig)
    cla
    if ndims(img) == 2
        colormap(gray(255))
    end
    imagesc(img/255)
    axis image; hold on
    
    % Plot just the positions
    U = [keypoints.U];
    plot(U(1,:), U(2,:), 'b.')
    if nargin == 3
        drawnow
        return
    end
    
    
    % Need to find the rotation matrices which rotates the sphere 
    % pole to the keypoints spherical location
    % Z-axis rotation is the keypoint rotation
    alpha = [keypoints.rot];         % Z-axis
    S = [keypoints.S];      
    beta = acos(S(3,:));             % Y-axis
    gamma = atan2(S(2,:),S(1,:));   % Z-axis
    
    % Support region sizes (angles psi)
    psi = sqrt(2) * 10 * sqrt([keypoints.kt]);
    
    
    % Draw scale and rotation as vector
    %================================================================
    if strcmp(keystyle,'vector')
        side_angle = 5*pi/6;
        np = 10;
        for n = 1:length(keypoints)

            % Rotation from pole to spherical coord
            R = euler_matrix('z',gamma(n)) * euler_matrix('y',beta(n)) * ...
                                             euler_matrix('z',alpha(n))';
                                     
            % Set up the arrow line on the sphere
            theta = linspace(0, psi(n), np);
            Sarrow = R * [sin(theta); zeros(1,length(theta)); cos(theta)];
        
            % Map back to the image plane and plot line
            u = (vars.m + 1) * [Sarrow(1,:)./(Sarrow(3,:)+1); 
                            Sarrow(2,:)./(Sarrow(3,:)+1)] + repmat(vars.pp,[1 np]);
            plot(u(1,:), u(2,:), '-r')
        
            % Plot the arrow
            arrow_scale = 2 * vars.m * sqrt(keypoints(n).kt);
            arrow_rot = atan2(u(2,end)-u(2,end-1), u(1,end)-u(1,end-1));
            h1X = u(1,end) + arrow_scale * cos(arrow_rot + side_angle);
            h1Y = u(2,end) + arrow_scale * sin(arrow_rot + side_angle);
            h2X = u(1,end) + arrow_scale * cos(arrow_rot - side_angle);
            h2Y = u(2,end) + arrow_scale * sin(arrow_rot - side_angle);
            plot([u(1,end) h1X u(1,end) h2X],[u(2,end) h1Y u(2,end) h2Y], '-r');
        end
    end
        
    
    % Draw the support region
    %================================================================
    if strcmp(keystyle,'support')
        np = 50;
        xunit = cos(linspace(0,2*pi,np));
        yunit = sin(linspace(0,2*pi,np));
        for n = 1:length(keypoints)

            % Rotation from pole to spherical coord
            R = euler_matrix('z',gamma(n)) * euler_matrix('y',beta(n)) * ...
                                             euler_matrix('z',alpha(n))';
                                     
            % Set up the support region on sphere
            theta = repmat(psi(n),[1 np]);
            Ssupport = R * [sin(theta).*xunit; sin(theta).*yunit; cos(theta)];
            
            % Map back to the image plane and plot line
            u = (vars.m + 1) * [Ssupport(1,:)./(Ssupport(3,:)+1); 
                    Ssupport(2,:)./(Ssupport(3,:)+1)] + repmat(vars.pp,[1 np]);
            plot(u(1,:), u(2,:), '-r')
        end
    end
end




        
%         % Set up the circular support region on the sphere
%         theta = sqrt(2) * 8 * f(i).kt * ones(1,50);
%         phi = linspace(0, 2*pi, 50);
%         Sx = sin(theta) .* cos(phi);
%         Sy = sin(theta) .* sin(phi);
%         Sz = cos(theta);
%         S_new = R * [Sx; Sy; Sz];
%         
%         % Map back to the image plane
%         theta = asin(sqrt(sum(S_new(1:2,:).^2, 1)));
%         phi = -atan2(S_new(2,:) , S_new(1,:));
%         R_im = (cam.m + 1) * tan(theta/2);
%         u = R_im .* cos(phi) + cam.Cx;
%         v = R_im .* sin(phi) + cam.Cy;
%         plot(u, v, '-g')
%     end


% Euler matricies
% oops, I think these are left-hand...
function A = euler_matrix(axis,theta)

switch axis
    case 'x'
        A = [1      0           0;
             0      cos(theta)  -sin(theta);
             0      sin(theta)   cos(theta)];
         
    case 'y'
        A = [cos(theta)     0   sin(theta);
             0              1   0;
             -sin(theta)    0   cos(theta)];
         
    case 'z'
        A = [cos(theta)     -sin(theta) 0;
             sin(theta)     cos(theta)  0;
             0              0           1];
         
    otherwise
        error('Enter an axis')
end
end






    
