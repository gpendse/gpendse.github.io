function z = projected_mog_ica_test( nruns, outputfile )
%=========================================================================
%   COPYRIGHT NOTICE AND LICENSE INFO:
%
%   * Copyright (C) 2010, Gautam V. Pendse, McLean Hospital
%   
%   * Matlab implementation of the simulations on synthetic data as described in the paper:
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
%
%=========================================================================
%   INPUTS:
%
%       =>       nruns = how many times should PMOG and FastICA be run for same sources but different mixing matrix
%       =>  outputfile = a .mat file in which all output will be saved   
%=========================================================================
%
%=========================================================================
%   PURPOSE:
%   
%   * Can projected mog give good BSS performance? To answer this question, we proceed as follows:
%
%   (1) Generate MOG sources
%   (2) Mix them with a random matrix
%   (3) Unmix using FastICA (defl) and PMOG
%   (4) Repeat (2)-(3) many times (say 100) for the same MOG sources.
%   (5) For each run, for both FastICA and PMOG, run raicar and compute correlation coefficient of the best match component from 
%       FastICA and PMOG with each true source.
%   (6) At the end of this procedure, you should be able to generate a plot where on the x-axis you have true source number and on 
%       the y-axis you display a box plot of the bestmatch corrcoef from FastICA and PMOG over 100 runs (say).
%   (7) If PMOG, is more efficient, the correlation coefficents should be higher than those for FastICA for all sources   
%=========================================================================


%=========================================================================
    % how many MOG sources do you want
    nsources = 7;

    % first generate MOG sources
    cmsm = create_mog_source_mixture( nsources, 20, 1000 ); % 4 mog sources, A can be 20 by 4
    
    % get the sources
    S = cmsm.S;
%=========================================================================

%=========================================================================
    % initialize counter
    count = 1;
    
    while ( count <= nruns )
        
       % generate random mixing matrix
       A = rand( 20, nsources );
       
       % generate mixed data (20 by 1000)
       X = A * S;
       
       % run FastICA
       fica = fastica( X, 'numofIC', nsources, 'g', 'tanh', 'approach', 'defl' );
       
       % run raicar on S and fica
       mat_array_fica{1} = S;
       mat_array_fica{2} = fica;
       
       rcr_fica = raicar_type_sorting_mat( mat_array_fica, 0 );
        
       
       % compute best match corrcoef in fica for each source
       for i = 1:nsources
          temp = corrcoef( rcr_fica.HC{i}(1,:), rcr_fica.HC{i}(2,:) );
          best_match_fica_for_source( rcr_fica.M_calc(1,i), count ) = temp(1,2); % each row is the best match corrcoef for one source
       end
        
        % run pmog        
        pmi = ppca_mm_ica( X + 1e-6*randn(size(X,1),size(X,2)), 1, nsources ); 
        
        % close all figs
        close all;
        
        % run raicar on S and pmog.final_source_estimates
        mat_array_pmog{1} = S;
        mat_array_pmog{2} = pmi.final_source_estimates;
        
        rcr_pmog = raicar_type_sorting_mat( mat_array_pmog, 0 ); 
        
        % compute best match corrcoef in pmog for each source
        for i = 1:nsources
           temp = corrcoef( rcr_pmog.HC{i}(1,:), rcr_pmog.HC{i}(2,:) );
           best_match_pmog_for_source( rcr_pmog.M_calc(1,i), count ) = temp(1,2); % each row is the best match corrcoef for one source
        end
       
       
        % update counter
        count = count + 1;
        
    end

    z = [];
    save(outputfile);
    %keyboard;
%=========================================================================


end