%% How to run PMOG based BSS on a given dataset?
% This demo program will illustrate PMOG based BSS on sources generated using the FastICA package.
function z = pmog_demo1()
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

%% (1) Generate sources and their mixture
%% 
% * We will call the function demosig.m from the FastICA package (http://www.cis.hut.fi/projects/ica/fastica/).
% * For convenience FastICA package is included with the distribution of PMOG code.
% 

    %=========================================================================
    % call demosig.m from FastICA package
    [sig,mixedsig]=demosig();
    
    % get sources
    S = sig;
    
    % get mixed signals
    X = mixedsig;
    
    % how many sources do we have
    nsources = size(S,1);
    %=========================================================================
    
       
%% (2) Do PMOG based BSS
% Please note that:
%%
% 
% * PMOG fits a mixture of PPCA model before BSS and hence it expects a non-square and noisy input.
% * Since our input is without noise, we add a small amount of noise to make the PPCA mixture estimation stable.
        
        % run pmog        
        pmi = ppca_mm_ica( X + 1e-6*randn(size(X,1),size(X,2)), 1, nsources ); 
        
%% (3) Check results        
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

%% (4) Visualize results
% Here we plot the original sources and the unmixed sources using PMOG

%%
% *Original sources*
        % first plot original sources
        fh1 = figure('color',[1,1,1]);set(gca,'FontSize',12);
        for i = 1:nsources
           subplot(2,2,i);plot( sig(i,:),'r-','LineWidth',1 );
           legend(['source ',num2str(i)]); 
        end

        
%%
% *Mixed signals*
        
        % next plot the mixed signals
        fh2 = figure('color',[1,1,1]);set(gca,'FontSize',12);
        for i = 1:nsources
           subplot(2,2,i);plot( mixedsig(i,:),'c-','LineWidth',1 );
           legend(['mixed signal ',num2str(i)]); 
        end
        
        
%%
% *PMOG recovered sources*
        
        % finally plot the recovered sources
        fh3 = figure('color',[1,1,1]);set(gca,'FontSize',12);
        for i = 1:nsources
           subplot(2,2,i);plot( pmi.final_source_estimates(i,:),'b-','LineWidth',1 );
           legend(['PMOG source ',num2str(i)]); 
        end
        
end