%% How to run PMOG based BSS on a given dataset?
% This demo program will illustrate PMOG based BSS using an image dataset from the Berkeley Segmentation Dataset and Benchmark (BSD):
% http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/. This code will look for a file called ../data/image_pmog_data2.mat
function z = pmog_demo3()
%% Copyright Notice
%=========================================================================
%   COPYRIGHT NOTICE AND LICENSE INFO:
%
%   * Copyright (C) 2010, Gautam V. Pendse, McLean Hospital
%   
%   * Matlab demo of the PMOG algorithm described in:
%   
%       Gautam V. Pendse, "PMOG: The projected mixture of Gaussians model with application to blind source separation", 
%       arXiv:1008.2743v1 [stat.ML, cs.AI, stat.ME], 46 pages, 9 figures, 2010. [5.5 MB] 
%
%   * License: Please see the file license.txt included with the code distribution
%              
%   
%   * If you use this code in your own work or alter this code, please cite the above paper. This code is distributed "as is" and no 
%   guarantee is given as to the accuracy or validity of the results produced by running this code. No responsibility will be assumed 
%   of the effects of running this code or using its results in any manner.
%
%   * AUTHOR: GAUTAM V. PENDSE
%     DATE: 2nd April 2010
%=========================================================================

%% (1) Get sources from Berkeley Segmentation Dataset and Benchmark
% Read the image data from ../data/image_pmog_data2.mat

    %=========================================================================
    % load .mat file containing BSD images
    matfile = load('../data/image_pmog_data2.mat');
    
    % get the raw images from matfile structure
    S = matfile.s; % 3 by 154401, each image is 481 by 321 in size    
    %=========================================================================
    
%% (2) Create mixed data using a random mean vector and mixing matrix

    %=========================================================================
    % get the mean vector
    mu = matfile.mumix;
    
    % get mixing matrix
    A = matfile.Amix;
    
    % create mixed data
    X = repmat(mu,1,size(S,2)) + A * S;
    
    % how many MOG sources do we have
    nsources = 3;
    %=========================================================================
    
       
%% (3) Do PMOG based BSS
% Please note that:
%%
% 
% * PMOG fits a mixture of PPCA model before BSS and hence it expects a non-square and noisy input.
% * Since our input is without noise, we add a small amount of noise to make the PPCA mixture estimation stable.
% * In this case, we will not enforce orthogonality on projection vectors by calling ppca_mm_ica_no_orth.m        

        % run pmog        
        pmi = ppca_mm_ica_no_orth( X + 1e-6*randn(size(X,1),size(X,2)), 1, nsources ); 
        
%% (4) Check results        
%%
% 
% * Scroll up to see the final PMOG estimated densities of blind sources.
% * We will compute correlation coefficients between the true sources and
% the estimated sources after a "component matching" step (see below).
% * The "component matching" step includes a "sign-flipping" step to
% account for sign indeterminacy in BSS.
% 
        
        % run raicar on S and pmog.final_source_estimates
        mat_array_pmog{1} = S;
        mat_array_pmog{2} = pmi.final_source_estimates;
        
        % call raicar_type_sorting_mat.m which will match the two sets of sources in the best possible way including sign-flipping
        rcr_pmog = raicar_type_sorting_mat( mat_array_pmog, 0 ); 
        
        % compute best match corrcoef in pmog for each source
        for i = 1:nsources
           temp = corrcoef( rcr_pmog.HC{i}(1,:), rcr_pmog.HC{i}(2,:) );
           best_match_pmog_for_source( rcr_pmog.M_calc(1,i), 1 ) = temp(1,2); % each row is the best match corrcoef for one source
        end
       
        fh = figure('color',[1,1,1]);set(gca,'FontSize',12); 
        plot( 1:nsources, best_match_pmog_for_source, 'ro-', 'MarkerSize',6,'MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor','b' );
        xlabel('source number ');ylabel('best match corrcoef in PMOG based BSS');


%% (5) Visualize results
% Here we display the original source images, the mixed images and the recovered images

%%
% *Original images*

        % first show original images
        fh1 = figure('color',[1,1,1]);set(gca,'FontSize',12);
        for i = 1:nsources
           subplot(1,3,i);imagesc( reshape( S(i,:), 481, 321 ) );
           title( ['source ',num2str(i)] );
        end

%%
% *Mixed images*

        % next show mixed images
        fh1 = figure('color',[1,1,1]);set(gca,'FontSize',12);
        for i = 1:nsources
           subplot(1,3,i);imagesc( reshape( X(i,:), 481, 321 ) );
           title( ['mixed image ',num2str(i)] );
        end

        
        
%%
% *PMOG recovered images*

        % finally show the recovered images
        fh1 = figure('color',[1,1,1]);set(gca,'FontSize',12);
        for i = 1:nsources
           subplot(1,3,i);imagesc( reshape( pmi.final_source_estimates(i,:), 481, 321 ) );
           title( ['PMOG image ',num2str(i)] );
        end
                
        
end