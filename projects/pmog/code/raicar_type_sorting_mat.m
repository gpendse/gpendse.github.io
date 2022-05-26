function z = raicar_type_sorting_mat( mat_array, output_flag )
%=========================================================================
%   COPYRIGHT NOTICE AND LICENSE INFO:
%
%   * Copyright (C) 2010, Gautam V. Pendse, McLean Hospital
%   
%   * Helper algorithm for Projected MOG based BSS:
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
%     DATE: 11th Feb 2010
%=========================================================================
%
%=========================================================================
%               PURPOSE:
%  Given K 4-D images of the same size, align them with each other using
%  a RAICAR type analysis. This can be used for aligning the ICA components
%  for a dataset across multiple ICA runs for subsequent averaging and
%  reporting. 
%  
%  We implement the algorithm by Yang et. al (2008):
%  Yang, Z. and LaConte, S. and Weng, X. and Hu, X. 
% "Ranking and Averaging Independent Component Analysis by Reproducibility (RAICAR)", Human Brain Mapping, 29:711?725 (2008) 
%=========================================================================
%
%=========================================================================
%               INPUTS:
%  mat_array = K by 1 cell array of matrices each of size nC by n (each row is one component of length n). K = number of runs of the algorithm.
%  output_flag = produce output or not? output_flag = 1 produces figures (default)
%=========================================================================
%
%=========================================================================
%               OUTPUTS:
%
% z.M_calc = Matrix such that Row i is indices of components found in Run i
%            and each column is one aligned component
%
% z.AC = Cell array of length nC such that AC{r} will be a K by 2 matrix whose first column will be the run number and the second
%        column will be the component number of the rth matched component
%
% z.AC_sorted = same as AC but sorted by the first column of AC{r} such that AC{r} always has first column (1:K) for all r
%
% z.reproducibility_nocutoff = RAICAR based reproducibility calculation for observed components (without using a cutoff)
%
% z.reproducibility_nocutoff_normalized = same as reproducibility_nocutoff but normalized by K*(K-1)/2
%
% z.HC = cell array such that HC{c} = K by n matrix each row of which is a realization of the cth aligned component across K runs
%
% z.reproducibility_nocutoff_normalized_random = RAICAR applied to permuted CRCM
%
% z.reproducibility_cutoff_95_perc = 95% cutoff on calculated reproducibility of randomly permuted CRCM
%
% (1) AC{1} will be a K by 2 matrix whose first column will be the run number and the second
%     column will be the component number of matched components. For
%     example, if AC{1} = [1,3
%                          2,4
%                          3,7]
%     Then it means in aligned component 1, (Run 1, Comp 3), (Run 2, Comp 4) and (Run 3, Comp 7) match with each other!
%
% (2) AC_sorted is the same as AC except that the first column in each matrix AC_sorted{i} is now always (1:K)
%
% (3) In M_calc, Row i is indices of components found in Run i and each column is one aligned component. In other words: 
%     M_calc(i,:) = component indices from Run i. 
%     M_calc(:,j) = one aligned component across Runs 1:K
%     M_calc(i,j) = Component index from Run i that is aligned with M_calc(:,j) [the jth aligned component]
%
% (4) error_flag = 1 if an error occured during processing and 0 on no error.
%
% (5) reproducbility = sum of cross-correlation coefficients among aligned components thresholded at a user specified level 
%                      based on histogram of all cross-correlation coefficients (per Yang et. al)
%
% (6) reproducibility_normalized = reproducibility / ((K-1)*K/2). Here K*(K-1)/2 is the max possible value of reproducibility in K runs
%
% (7) HC = cell array such that HC{c} = K by n matrix each row of which is a realization of the cth aligned component across K runs

%=========================================================================


%=========================================================================
%               Check input args
    if ( nargin < 1 )
        disp('Usage: z = raicar_type_sorting_mat( mat_array, output_flag )');
        z = [];
        return;
    end
    
    if ( nargin == 1 )
        output_flag = 1;
    end
%=========================================================================

%=========================================================================
%               add required paths
    addpath('/home/gpendse/MATLAB/KSMOOTH');
%=========================================================================

%=========================================================================
%               How many mat arrays's do we have
    K = length( mat_array );
%=========================================================================

%=========================================================================
%               Make sure all elements of mat array have the same size
    [nC, n] = size( mat_array{1} );
    
    for r = 2:K
       if ( size( mat_array{r}, 1 ) ~= nC || size( mat_array{r}, 2 ) ~= n )
           disp('Each element of mat_array should have the same size!');
           z = [];
           return;
       end        
    end
%=========================================================================


%=========================================================================
% Get all 4-D images as matrices nC by n into a cell array

    % now load data into a cell array R of length K
    for r = 1:K
       R{r} = mat_array{r}; % nC by n [each row is one component]
    end
%=========================================================================

%=========================================================================
% Get K*nC by K*nC cross-correlation matrix. This matrix can also be
% thought of as a K by K block matrix with each block of size nC by nC.
% The block at position (i,j) is the cross-correlation matrix between nC
% components from run i and run j.

    % initialize matrix
    G = zeros( K*nC, K*nC );
    
    % populate matrix G with ABSOLUTE spatial corr-coeffs
    for i = 1:K
        for j = 1:K
            fprintf('\r%s\n',['creating block matrix (',num2str(i),',',num2str(j),')']);
            gsm = get_scc_matrix( R{i}, R{j} ); % nC by nC matrix
            G( (i-1)*nC + 1 : i*nC, (j-1)*nC + 1 : j*nC ) = abs( gsm.scc ); % (i,j)th block of G of size nC by nC
        end
    end
    
    
    % save a copy of G in case you need it later
    G_copy = G;
    
    fprintf('\n');
%=========================================================================

%=========================================================================
    % do component matching
    mcp = match_components( nC, K, G );
    M_calc = mcp.M_calc;
    AC = mcp.AC;
    AC_sorted = mcp.AC_sorted;
    reproducibility_nocutoff = mcp.reproducibility_nocutoff;
    reproducibility_nocutoff_normalized = mcp.reproducibility_nocutoff_normalized;
    
    z.M_calc = M_calc;
    z.AC = AC;
    z.AC_sorted = AC_sorted;
    z.reproducibility_nocutoff = reproducibility_nocutoff;
    z.reproducibility_nocutoff_normalized = reproducibility_nocutoff_normalized;
%=========================================================================

%=========================================================================
%   % finally output a cell array HC of length nC such that:
%      HC{1} = K by n matrix each row of which is a realization of an aligned component
%   % The rows have also been sign flipped to match the 1st row
    for c = 1:nC
        HC{c} = [R{1}( M_calc(1,c), : )]; % 1st run, M_calc(1,c)th row corresponding to the cth aligned component (1 by n)
        
        for r = 2:K
            
           tempC = R{r}( M_calc(r,c), : ); % rth run, M_calc(r,c)th row corresponding to the cth aligned component (1 by n)            
           testcrcf = corrcoef( tempC', HC{c}(1,:)' );
           
           if ( testcrcf(1,2) > 0 )
              % no need to flip 
              HC{c} = [HC{c}; tempC];
           else
              disp('sign flipping!');
              % we need to flip tempC before adding it to HC{c}
              HC{c} = [HC{c}; -tempC];
           end
                      
        end        
    end
    
    % save in output structure
    z.HC = HC;
%=========================================================================


%=========================================================================
%   randomly permute rows and cols of G_copy and call match_components.
%   Store the maximal reproducibility for this permutation. Repeat this
%   process many times and draw a histogram of reproducibility under random
%   permutation of the rows of G_copy
    nrand_matches = 100; % how many random matches do you want?
    
    reproducibility_nocutoff_normalized_random = [];
    
    for m = 1:nrand_matches
        
        rand_perm_vec = randperm( size(G_copy,1) );
        
        mcp_rand = match_components( nC, K, G_copy(rand_perm_vec, rand_perm_vec) );
        
        reproducibility_nocutoff_normalized_random = [reproducibility_nocutoff_normalized_random, mcp_rand.reproducibility_nocutoff_normalized];
        
    end
    z.reproducibility_nocutoff_normalized_random = reproducibility_nocutoff_normalized_random;
    
    z.reproducibility_cutoff_95_perc = quantile( reproducibility_nocutoff_normalized_random, 0.95 );
%=========================================================================

%=========================================================================
    % Make figures here
    if ( output_flag == 1 )
        fh1 = figure('Visible','on','color',[1,1,1]);orient(fh1,'landscape');set(fh1,'Units','inches','Position',[1,1,10,8]);

        subplot(2,1,1);
        hist( z.reproducibility_nocutoff_normalized_random, 100);
        set(gca,'LineWidth',1,'FontSize',12);
        hist_handle = get(gca, 'Children');
        set(hist_handle(1),'FaceColor','c');
        xlabel('Normalized Reproducibility');
        ylabel('Histogram (randomly permuted cross-correlation matrix)');
        title( ['Reproducibility cutoff 95 perc = ',num2str(z.reproducibility_cutoff_95_perc)] );
        
        subplot(2,1,2);
        bar( z.reproducibility_nocutoff_normalized, 'g');
        hold on;plot( z.reproducibility_cutoff_95_perc*ones(length(z.reproducibility_nocutoff_normalized)), 'color','r', 'LineWidth', 1 );
        set(gca,'LineWidth',1,'FontSize',12);
        xlabel('Component number');
        ylabel('Normalized reproducibility'); 
        
    end
%=========================================================================


% %=========================================================================
%     % determine cutoff on cross-correlation among aligned components
%     tempG = triu( G_copy, 1 );
%     tempG = tempG( find(tempG ~= 0) );
%     
%     % store all pairwise corr coeffs here
%     z.pairwise_corrcoefs = tempG;
%     
%     % get frequency count using 100 bins between 0 and 1
%     [Nh, Xh] = hist( tempG, 100 );
%     
%     % do kernel smoothing for a range of lambda values and use the best one found using cross validation
%     
%     %   NOTE: we use log(Nh) instead of Nh in the command below as per the RAICAR paper
%     %   oks = ks( Xh, log(Nh + eps), linspace(0.005,0.1,100) );
%     oks = ks( Xh, Nh, linspace(0.01,0.2,100) );
% 
%         
%     % find the minimum on the smoothed curve
%     fmav = find_mins_along_vector( oks.y_fit );
%     % save fmav for future use
%     z.fmav = fmav;
%     
%     % excluding the possibility that minimum occurs at the first and last index where is the minimum located?
%     good_indices = find( fmav.min_locations ~= 1 & fmav.min_locations ~= length(oks.y_fit) );
%         
%     if ( isempty(good_indices) ~= 1 )
%         min_index = find( oks.y_fit == min(fmav.min_values(good_indices)) );
%     else
%         min_index = find( oks.y_fit == min(oks.y_fit) );
%     end
%     min_index = min_index(1);
%     z.min_index = min_index;
%     
%     % find the cross-correlation cutoff as the x-value at this minimum
%     crsc_cutoff = oks.x ( min_index );
%     
%     % save oks in z for future use
%     z.oks = oks;
%     
%     % save crsc cutoff
%     z.crsc_cutoff = crsc_cutoff;
% %=========================================================================

% %=========================================================================
%     % Calculate K by K cross correlation matrix among aligned components.
%     % There will  be nC such K by K matrices, one for each distinct aligned
%     % component.
%     for c = 1:nC
%         % store index into G of aligned components here
%         index_c = [];
%         for r = 1:K
%             index_c = [index_c; (r-1)*nC + M_calc(r,c)]; % Run r, component number M_calc(r,c) for cth aligned component
%         end
%         CRCM{c} = G_copy( index_c, index_c ); % K by K cross-correlation matrix among aligned components
%     end
% %=========================================================================
% 
% 
% %=========================================================================
%     % Calculate reproducibility index for each aligned component
%     for c = 1:nC
%         tempCRCM = triu( CRCM{c}, 1 );
%         tempCRCM = tempCRCM( find(tempCRCM >= crsc_cutoff) );
%         % If all elements of tempCRCM (part above the main diagonal of
%         % CRCM{c}) are > crsc_cutoff then length(tempCRCM) will be 1 + 2 +
%         % ..+(K-1) = (K-1)*K/2 elements. Since each element can have a
%         % maximum absolute correlation coefficient of 1. sum(tempCRCM) can
%         % have a maximum value of (K-1)*K/2
%         reproducibility(c) = sum( tempCRCM );
%     end
%     z.reproducibility = reproducibility;
%     
%     % normalize reproducibility by the maximum possible value of (K-1)*K/2
%     z.reproducibility_normalized = reproducibility / ((K-1)*K/2); 
% %=========================================================================



% %=========================================================================
% %   do random component matching to see what reproducibilities you get by random matching
%     nrand_matches = 100; % how many random matches do you want?
%     
%     for m = 1:nrand_matches
% 
%         for i = 1:K
%             component_index{i} = (1:nC)'; % each of the K runs gets an index 1:nC
%         end
%         
%         for c = 1:nC
%             index_c_rnd = [];
%             for r = 1:K
%                 
%                 rand_comp_from_run_r = randsample( component_index{r}, 1 );
%                 index_c_rnd = [index_c_rnd; (r-1)*nC + rand_comp_from_run_r];
%                 component_index{r} = setdiff( component_index{r}, rand_comp_from_run_r );
%             end
%             CRCM_rnd{c} = G_copy( index_c_rnd, index_c_rnd ); % K by K cross-correlation matrix among randomly aligned components
%             tempCRCM_rnd = triu( CRCM_rnd{c}, 1 );
%             tempCRCM_rnd = tempCRCM_rnd( find(tempCRCM_rnd >= 0) );
% 
%             reproducibility_random_nocutoff(c, m) = sum(tempCRCM_rnd);
% 
%         end
%     
%     end
%     z.reproducibility_random_nocutoff_normalized = reproducibility_random_nocutoff/(K*(K-1)/2);
% %=========================================================================


% %=========================================================================
% %   do random component matching to see what reproducibilities you get by random matching
%     nrand_matches = 100; % how many random matches do you want?
%     
%     for m = 1:nrand_matches
% 
%         for i = 1:K
%             component_index{i} = (1:nC)'; % each of the K runs gets an index 1:nC
%         end
%         
%         for c = 1:nC
%             index_c_rnd = [];
%             for r = 1:K
%                 
%                 rand_comp_from_run_r = randsample( component_index{r}, 1 );
%                 index_c_rnd = [index_c_rnd; (r-1)*nC + rand_comp_from_run_r];
%                 component_index{r} = setdiff( component_index{r}, rand_comp_from_run_r );
%             end
%             CRCM_rnd{c} = G_copy( index_c_rnd, index_c_rnd ); % K by K cross-correlation matrix among randomly aligned components
%             tempCRCM_rnd = triu( CRCM_rnd{c}, 1 );
%             tempCRCM_rnd = tempCRCM_rnd( find(tempCRCM_rnd >= 0) );
% 
%             reproducibility_random_nocutoff(c, m) = sum(tempCRCM_rnd);
% 
%         end
%     
%     end
%     z.reproducibility_random_nocutoff_normalized = reproducibility_random_nocutoff/(K*(K-1)/2);
% %=========================================================================


% %=========================================================================
%     % Make figures here
%     if ( output_flag == 1 )
%         fh1 = figure('Visible','on','color',[1,1,1]);orient(fh1,'landscape');set(fh1,'Units','inches','Position',[1,1,10,8]);
% 
%         subplot(2,2,1);
%         plot( oks.x, oks.y, 'color','b','LineWidth',1);hold on;
%         plot( oks.x, oks.y_fit, 'color','r','LineWidth',1 );
%         plot( oks.x(fmav.min_locations), oks.y_fit(fmav.min_locations), 'Color','c','LineWidth',1,'LineStyle','none','Marker','o','MarkerFaceColor','c');
%         plot( crsc_cutoff, oks.y_fit(min_index), 'Color','m','LineWidth',1,'LineStyle','none','Marker','o','MarkerFaceColor','m');
%         set(gca,'LineWidth',1,'FontSize',12);
%         xlabel(' Correlation Coefficient ');
%         ylabel(' Frequency Count ');
%         legend('Raw','Smoothed','Minima','Best minimum','Location','Best');
% 
%         subplot(2,2,2);
%         plot( oks.lambda, log(oks.CVE), 'color','r','LineWidth',1 );hold on;
%         plot( oks.lambda(oks.q), log(oks.CVE(oks.q)), 'color','g','Marker','o','MarkerFaceColor','g' );
%         set(gca,'LineWidth',1,'FontSize',12);
%         xlabel(' Lambda ');
%         ylabel(' Log(CVE) ' );
%         legend('CVE curve','Minimum','Location','Best');
% 
%         subplot(2,2,3);
%         bar( z.reproducibility_nocutoff_normalized, 'g');
%         hold on;plot( 0.5*ones(length(z.reproducibility_nocutoff_normalized)), 'color','r', 'LineWidth', 1 );
%         set(gca,'LineWidth',1,'FontSize',12);
%         xlabel('Component number');
%         ylabel('Normalized reproducibility');    
%     
%         subplot(2,2,4);
%         hist( z.reproducibility_random_nocutoff_normalized(:), 100);
%         set(gca,'LineWidth',1,'FontSize',12);
%         xlabel('Reproducibility');
%         ylabel('Histogram (random matching)');
%     
%     end
% %=========================================================================
    

end


function o = match_components( nC, K, G )
%=========================================================================
%               AUTHOR: GAUTAM V. PENDSE
%               DATE: 4th Mar 2010
%=========================================================================
%
%=========================================================================
%               PURPOSE:
%  Execute the RAICAR algorithm here
%=========================================================================
%
%=========================================================================
%               INPUTS:
%              nC = number of components
%               K = number of runs
%               G = (nC*K) by (nC*K) cross correlation (absolute) matrix
%=========================================================================
%
%=========================================================================
%               OUTPUTS:
%             o.AC = a cell array of nC elements, each an array of size K by 2
%                    giving the Run and Comp numbers of match for each aligned component.
%               
%      o.AC_sorted = a cell array of nC elements after sorting each cell array in AC by run number (or first columns)
%
%         o.M_calc = a matrix such that Row i is Run i and each column is one aligned component
%
%     o.error_flag = 0 if o.M_calc passes row checks and 1 otherwise
%
%     o.reproducibility_nocutoff = sum of absolute cross-correlations between aligned components for each aligned component
%
%     o.reproducibility_nocutoff_normalized = o.reproducibility_nocutoff / (K*(K-1)/2)
%
%=========================================================================


%=========================================================================
%   first save a copy of G
    % populate the block diagonals with zeros (since components from the same run cannot be matched with each other)
    for i = 1:K
        for j = i:i
            G( (i-1)*nC + 1 : i*nC, (j-1)*nC + 1 : j*nC ) = 0; % set (i,i)th block of G of size nC by nC = 0
        end
    end
    
    G_copy = G;
%=========================================================================

%=========================================================================
% Start component matching
    for q = 1 : nC

        disp(['forming aligned component ',num2str(q)]);
        % get row and col indices of the maximum element in G
        [mr, ms] = find( G == max(G(:)) );    
        mr = mr(1);
        ms = ms(1);
        
        % create a cell array of aligned components, AC. AC{1} will be a K by
        % 2 matrix whose first column will be the run number and the second
        % column will be the component number of matched components. For
        % example, if AC{1} = [1,3
        %                      2,4
        %                      3,7]
        % Then it means in aligned component 1, (Run 1, Comp 3), (Run 2, Comp
        % 4) and (Run 3, Comp 7) match with each other!
    
        % get Run number and Component number of match
        mgr = map_Gindex_to_RunAndComp( mr, ms, nC );
        
        AC{q} = [ mgr.i, mgr.M; mgr.j, mgr.N ]; % Run i, Comp M matches with Run j, Comp N
    
        % see how good the match is between Run i, Comp M and Run u, all comps where u not equal to (i and j)        
        for u = setdiff( 1:K, [mgr.i, mgr.j] )
           % search in G, row mr of block col u 
           G_prime = zeros( size(G) );
           G_prime( mr, (u-1)*nC + 1 : u*nC ) = G( mr, (u-1)*nC + 1 : u*nC );
           
           % get index of maximum
           [mru, msu] = find( G_prime == max(G_prime(:)) );
           mru = mru(1);
           msu = msu(1);

           % map index to run and comp number
           mgru = map_Gindex_to_RunAndComp( mru, msu, nC );
           
           % store match corrcoef and run/comp no. of match
           match_iM(u) = G_prime( mru, msu );
           run_comp_match_iM{u} = [mgru.j, mgru.N];% match across cols    
           run_index_iM{u} = [mru, msu]; % row and col index of match
        end
        
        % see how good the match is between Run j, Comp N and Run u, all comps where u not equal to (i and j)        
        for u = setdiff( 1:K, [mgr.i, mgr.j] )
           % search in G, col ms of block row u 
           G_prime = zeros( size(G) );
           G_prime( (u-1)*nC + 1 : u*nC, ms ) = G( (u-1)*nC + 1 : u*nC , ms );
           
           % get index of maximum
           [mru, msu] = find( G_prime == max(G_prime(:)) );
           mru = mru(1);
           msu = msu(1);

           % map index to run and comp number
           mgru = map_Gindex_to_RunAndComp( mru, msu, nC );
           
           % store match corrcoef and run/comp no. of match
           match_jN(u) = G_prime( mru, msu );
           run_comp_match_jN{u} = [mgru.i, mgru.M];% match across rows
           run_index_jN{u} = [mru, msu]; % row and col index of match
        end
        
        % populate the list AC{q} with (K-2) additional entries
        for u = setdiff( 1:K, [mgr.i, mgr.j] )
           
            if ( match_iM(u) == match_jN(u) )
               % same component matched from Run u
               AC{q} = [AC{q}; run_comp_match_iM{u}];
               G( run_index_iM{u}(1), : ) = 0;
               G( :, run_index_iM{u}(1) ) = 0;

               G( :, run_index_iM{u}(2) ) = 0;
               G( run_index_iM{u}(2), : ) = 0;
               
               G( run_index_jN{u}(1), : ) = 0;
               G( :, run_index_jN{u}(1) ) = 0;

               G( :, run_index_jN{u}(2) ) = 0;
               G( run_index_jN{u}(2), : ) = 0;
               
            else
                
               % else iM and jM match different components from Run u 
               if ( match_iM(u) > match_jN(u) )
                  % choose the component from Run u that matches iM
                  AC{q} = [AC{q}; run_comp_match_iM{u}]; 
                  G( run_index_iM{u}(1), : ) = 0;
                  G( :, run_index_iM{u}(1) ) = 0;

                  G( :, run_index_iM{u}(2) ) = 0;
                  G( run_index_iM{u}(2), : ) = 0;
               else
                  % choose the component from Run u that matches jN
                  AC{q} = [AC{q}; run_comp_match_jN{u}]; 
                  G( run_index_jN{u}(1), : ) = 0;
                  G( :, run_index_jN{u}(1) ) = 0;

                  G( :, run_index_jN{u}(2) ) = 0;
                  G( run_index_jN{u}(2), : ) = 0;
               end
            end
            
        end
       
        % zero out row mr and column ms and row ms and column mr from G
        G( mr, : ) = 0;
        G( :, ms ) = 0;
        G( ms, : ) = 0;
        G( :, mr ) = 0;
        
    end    
%=========================================================================

%=========================================================================
%   At this point AC should have nC elements, each an array of size K by 2
%   giving the Run and Comp numbers of match for each aligned component.
    o.AC = AC;
%=========================================================================

%=========================================================================
%   Sort each cell array in AC by run number (or first columns)
    for r = 1:nC
        [temp1, temp2] = sort( AC{r}(:,1), 'ascend' ); % sorting by row number
        AC_sorted{r} = AC{r}( temp2, : );
    end
    o.AC_sorted = AC_sorted;
    % The first column in each matrix AC_sorted{i} is now always (1:K)
%=========================================================================

%=========================================================================
    % calculate component numbers (Run1 to RunK)
    % Row i is Run i and each column is one aligned component
    for r = 1:nC
        if ( length(AC_sorted{r}(:,2)) ~= K )
            keyboard;
        end
       M_calc(:,r) = AC_sorted{r}(:,2); % K rows and nC columns
    end
    o.M_calc = M_calc;
    
    
    % make sure that each row of M_calc has elements (1:nC)
    error_flag = 0;
    for s = 1:K
       if ( isempty( setdiff(M_calc(s,:), (1:nC)) ) == 1 && isempty( setdiff((1:nC), M_calc(s,:)) ) == 1 )
           % this is what we want
       else
           % this is bad
           error_flag = 1;
       end
    end
    
    if ( error_flag == 1 )
        disp('Error: Each row of M_calc should have elements from 1:nC!');
    else
        disp('Passed check on each row of M_calc!!');
    end
    
    o.error_flag = error_flag;
%=========================================================================

%=========================================================================
    % Calculate K by K cross correlation matrix among aligned components.
    % There will  be nC such K by K matrices, one for each distinct aligned
    % component.
    for c = 1:nC
        % store index into G of aligned components here
        index_c = [];
        for r = 1:K
            index_c = [index_c; (r-1)*nC + M_calc(r,c)]; % Run r, component number M_calc(r,c) for cth aligned component
        end
        CRCM{c} = G_copy( index_c, index_c ); % K by K cross-correlation matrix among aligned components
    end
%=========================================================================


%=========================================================================
    % Calculate reproducibility index for each aligned component without using a cutoff
    for c = 1:nC
        tempCRCM = triu( CRCM{c}, 1 );
        tempCRCM = tempCRCM( find(tempCRCM >= 0) );
        % If all elements of tempCRCM (part above the main diagonal of
        % CRCM{c}) are > crsc_cutoff then length(tempCRCM) will be 1 + 2 +
        % ..+(K-1) = (K-1)*K/2 elements. Since each element can have a
        % maximum absolute correlation coefficient of 1. sum(tempCRCM) can
        % have a maximum value of (K-1)*K/2
        reproducibility_nocutoff(c) = sum( tempCRCM );
    end
    o.reproducibility_nocutoff = reproducibility_nocutoff;
    
    % normalize reproducibility by the maximum possible value of (K-1)*K/2
    o.reproducibility_nocutoff_normalized = reproducibility_nocutoff / ((K-1)*K/2); 
%=========================================================================

end


function o = map_RunAndComp_to_Gindex( i, M, j, N, nC )
%=========================================================================
%               AUTHOR: GAUTAM V. PENDSE
%               DATE: 11th Feb 2010
%=========================================================================
%
%=========================================================================
%               PURPOSE:
%  (1) We are given that G is a K by K block matrix with each block of size nC by nC.
%  The (i,j)th block of G is an nC by nC matrix containing the
%  cross-correlation between components in Run i with components in Run j.
%
%  (2) Thus the [(i-1)*nC + M, (j-1)*nC + N] th element is the cross-correlation
%  coefficient between Run i, Component M with Run j, Component N.
%
%  (3) Given r = (i-1)*nC + M, s = (j-1)*nC + N and nC we would like to
%  recover, i,j, M and N.
%=========================================================================
%
%=========================================================================
%               INPUTS:
%        i,M, j,N = Run and Comp numbers as defined above in PURPOSE
%              nC = size of each block in G as defined above in PURPOSE
%=========================================================================
%
%=========================================================================
%               OUTPUTS:
%        o.r = row index into G
%
%        o.s = col index into G
%=========================================================================

%=========================================================================
%               check input args
    if ( nargin ~= 5 )
        disp('o = map_RunAndComp_to_Gindex( i, M, j, N, nC )');
        o = [];
        return;
    end
%=========================================================================

%=========================================================================
%               calculate r and s here
    
   r = (i-1)*nC + M;
   s = (j-1)*nC + N;
%=========================================================================

%=========================================================================
%               save outputs here
   o.r = r;
   o.s = s;
%=========================================================================

end

function o = find_mins_along_vector( v )
%==========================================================================
%               AUTHOR: GAUTAM V. PENDSE
%               PURPOSE: Search for minimums along a vector v
%==========================================================================

    % get length of v
    n = length(v);
    min_locations = [];
    min_values = [];
    
    for r = 1:n
       
        if ( r == 1 )
            if ( v(r) < v(r+1) )
                min_locations = [min_locations, r];
                min_values = [min_values,v(r)];
            end
        elseif ( r == n )
            if ( v(r) < v(r-1) )
                min_locations = [min_locations, r];
                min_values = [min_values,v(r)];
            end
        else
            if ( v(r) < v(r-1) && v(r) < v(r+1) )
                min_locations = [min_locations, r];
                min_values = [min_values,v(r)];
            end
        end
        
    end


    o.min_locations = min_locations';
    o.min_values = min_values';
    
end


function o = map_Gindex_to_RunAndComp( r, s , nC )
%=========================================================================
%               AUTHOR: GAUTAM V. PENDSE
%               DATE: 11th Feb 2010
%=========================================================================
%
%=========================================================================
%               PURPOSE:
%  (1) We are given that G is a K by K block matrix with each block of size nC by nC.
%  The (i,j)th block of G is an nC by nC matrix containing the
%  cross-correlation between components in Run i with components in Run j.
%
%  (2) Thus the [(i-1)*nC + M, (j-1)*nC + N] th element is the cross-correlation
%  coefficient between Run i, Component M with Run j, Component N.
%
%  (3) Given r = (i-1)*nC + M, s = (j-1)*nC + N and nC we would like to
%  recover, i,j, M and N.
%=========================================================================
%
%=========================================================================
%               INPUTS:
%        r, s = indices into matrix G as defined above in PURPOSE
%          nC = size of each block in G as defined above in PURPOSE
%=========================================================================
%
%=========================================================================
%               OUTPUTS:
%        o.i = Run i
%        o.M = Component M in Run i
%
%        o.j = Run j
%        o.N = component N in Run j
%=========================================================================

%=========================================================================
%               check input args
    if ( nargin ~= 3 )
        disp('o = map_Gindex_to_RunAndComp( r, s , nC )');
        o = [];
        return;
    end
%=========================================================================

%=========================================================================
%               calculate remainders here
    rem_r = rem( r, nC );
    rem_s = rem( s, nC );
    
    if ( rem_r == 0 )
        i = floor( r/nC );
        M = nC;
    else
        i = floor( r/nC ) + 1;
        M = rem_r;
    end
    
    
    if ( rem_s == 0 )
        j = floor( s/nC );
        N = nC;
    else
        j = floor( s/nC ) + 1;
        N = rem_s;
    end
%=========================================================================

%=========================================================================
%               save outputs here
   o.i = i;
   o.M = M;
   
   o.j = j;
   o.N = N;
%=========================================================================

end

function o = get_scc_matrix( A, B )
%=========================================================================
%               AUTHOR: GAUTAM V. PENDSE
%               DATE: 11th Feb 2010
%=========================================================================
%
%=========================================================================
%               PURPOSE:
%  Given p by n matrices A and B. Compute the correlation
%  coefficient of each row of A with each row of B.
%=========================================================================
%
%=========================================================================
%               INPUTS:
%        A = p by n matrix (each row can be thought of as an image or a component)
%        B = p by n matrix (each row can be thought of as an image or a component)
%=========================================================================
%
%=========================================================================
%               OUTPUTS:
%        o.scc(i,j) = corrcoef of A(i,:) with B(j,:) [ith component in A with jth component in B]
%=========================================================================

%=========================================================================
%               check input args
    if ( nargin ~= 2 )
        disp('Usage: o = get_scc_matrix( A, B )');
        o = [];
        return;
    end
%=========================================================================

%=========================================================================
%               check size of A and B
    [p,n] = size(A);
    [p1,n1] = size(B);
    
    if ( p1 ~= p || n1 ~= n )
        disp('A and B must have the same size!');
        o = [];
        return;
    end
%=========================================================================

%=========================================================================
%               compute the p by p correlation matrix here
    scc = zeros( p, p );
    for i = 1:p
        for j = 1:p
            crcf = corrcoef( A(i,:)', B(j,:)' ); % ith comp in A with jth comp in B
            scc(i,j) = crcf(1,2);
        end
    end
%=========================================================================

%=========================================================================
%               save output here
    o.scc = scc;
%=========================================================================

end

function o = read_func_file( funcfile, maskfile )
%=========================================================================
%               AUTHOR: GAUTAM V. PENDSE
%               DATE: 11th Feb 2010
%=========================================================================
%
%=========================================================================
%               PURPOSE:
%  Read a 4-D functional file using the specified inclusion mask 
%=========================================================================
%
%=========================================================================
%               INPUTS:
%        funcfile = a 4-D functional file (such as output of MELODIC melodic_IC.nii.gz)
%        maskfile = 3-D inclusion mask (all voxels ~=0 will be analyzed)
%=========================================================================
%
%=========================================================================
%               OUTPUTS:
%        o.data = xdim by ydim by zdim by tdim array of 4-D image data
%        o.mask = xdim by ydim by zdim by 1 array of 3-D mask
%=========================================================================


%=========================================================================
%               check input args
    if ( nargin ~= 2 )
        disp('Usage: z = read_func_file( funcfile, maskfile )');
        o = [];
        return;
    end
%=========================================================================


%=========================================================================
%                   read the functional image
    S = read_image(funcfile);

    xdim = S.xdim;
    ydim = S.ydim;
    zdim = S.zdim;
    tdim = S.tdim;
%=========================================================================

%=========================================================================
%                   read the mask if possible
    if strcmpi(maskfile,'') == 1 || isempty(maskfile) == 1
        % use wholebrain mask
        disp('using whole brain mask from first image');
        temp = reshape(S.img, xdim*ydim*zdim, tdim)'; % tdim by xdim*ydim*zdim

        mask = temp( ceil(tdim/2), :); % 1 by xdim*ydim*zdim

        clear temp;

    else
        M = read_image(maskfile);

        if (M.xdim ~= xdim | M.ydim ~= ydim | M.zdim ~= zdim | M.tdim ~= 1)
           disp('mask must be a 3D image and of the same size as functional images');
           z = [];
           return;
        end

        mask = reshape(M.img, xdim*ydim*zdim, 1)'; % 1 by xdim*ydim*zdim

        clear M;    
    end

    % reshape mask
    mask = reshape( mask, xdim, ydim, zdim ); % xdim by ydim by zdim
%=========================================================================

%=========================================================================
%                   extract data here
    data = reshape( S.img, xdim, ydim, zdim, tdim ); % xdim by ydim by zdim by tdim
%=========================================================================

%=========================================================================
%                   save outputs here
    o.data = data; % xdim by ydim by zdim by tdim
    o.mask = mask; % xdim by ydim by zdim
%=========================================================================

end