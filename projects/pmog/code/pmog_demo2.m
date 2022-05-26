%% How to run PMOG based BSS on a given dataset?
% This demo program will illustrate PMOG based BSS using a simple MOG source example.
function z = pmog_demo2()
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

%% (1) Generate some sources
% We will call create_mog_source_mixture.m to generate a set of MOG
% sources.

    %=========================================================================
    % how many MOG sources do you want
    nsources = 3;

    % first generate MOG sources
    cmsm = create_mog_source_mixture( nsources, 20, 1000 ); 
    
    % get the sources from structure cmsm
    S = cmsm.S;
    %=========================================================================
    
    
%% (2) Generate a random mixing matrix
       
       % generate random mixing matrix
       A = rand( 20, nsources );

       
%% (3) Create mixed data using the mixing matrix and MOG sources

       % generate mixed data (20 by 1000)
       X = A * S;
       
%% (4) Do PMOG based BSS
% Please note that:
%%
% 
% * PMOG fits a mixture of PPCA model before BSS and hence it expects a non-square and noisy input.
% * Since our input is without noise, we add a small amount of noise to make the PPCA mixture estimation stable.
        
        % run pmog        
        pmi = ppca_mm_ica( X + 1e-6*randn(size(X,1),size(X,2)), 1, nsources ); 
        
%% (5) Check results        
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

end