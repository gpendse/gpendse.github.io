function z = ppca_mm_ica( X, R, q )
%=========================================================================
%   COPYRIGHT NOTICE AND LICENSE INFO:
%
%   * Copyright (C) 2010, Gautam V. Pendse, McLean Hospital
%   
%   * Matlab implementation of the algorithm for PMOG based BSS as described in the paper:
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
%       => X = p by n matrix of n vectors in R^p
%       => R = number of PPCA components to estimate
%       => q = a scalar indicating the common rank for each PPCA component
%=========================================================================
%
%
%=========================================================================
%   PURPOSE:
%   
%   * First a PPCA mixture model will be fitted to input data. For more details on PPCA mixture model please see the paper:
%     
%       Tipping, M. E. and C. M. Bishop (1999b). Mixtures of probabilistic principal component analysers. 
%       Neural Computation  11(2), 443?482.
% 
%   * Next an orthogonal matrix W will be estimated such that: 
%   
%       => w_i^T * marginal_source_mean (from ppca_mm) is as non-Gaussian as possible. 
%
%   * This is done by calling projected_mog_ica.m on marginal_source_mean multiple times until the entire matrix W is estimated. 
%     W will be of size q by q.
%=========================================================================
%
%=========================================================================
%   OUTPUTS:
%              
%       =>                      z.W = q by q matrix containing projection vectors
%       =>                   z.ppca = structure returned by ppca_mm.m
%       => z.final_source_estimates = q by n matrix of final source estimates
%       =>    z.final_mixing_matrix = final mixing matrix if PPCA MM is fitted with only 1 PPCA mixture component
%=========================================================================


%=========================================================================
%               check input args
    if ( nargin ~= 3 )
        disp('Usage: z = ppca_mm_ica(X, R, q)');
        z = [];
        return;
    end
%=========================================================================

%=========================================================================
%               check input sizes
    [p, n] = size(X); % we will assume that n >> p
 
    if ( size(q,1) ~= 1 || size(q,2) ~= 1 )
        disp('q must be a scalar!');
        z = [];
        return;
    end
    
    if ( q > p )
        disp('q must be < p!');
        z = [];
        return;
    end        
%=========================================================================

%=========================================================================
%               first call ppca_mm
    ppca = ppca_mm( X, R, q*ones(1,R) );
    z.ppca = ppca;
%=========================================================================

%=========================================================================
%               next call projected_mog_ica
    % initialize W
    W = [];
    for k = 1:q
        % use 5 component mixture (should be sufficient for most
        % non-Gaussian densities)
        pmog{k} = projected_mog_ica( ppca.marginal_source_mean, 5, W );
        W = [W, pmog{k}.w];
        % The modified mixing matrices are given by ppca.A{k}*W
        z.pmog{k} = pmog{k};
    end
%=========================================================================

%=========================================================================
%               compute W' * ppca.marginal_source_mean 
%               (q by n matrix of final source estimates)
    z.W = W;
    z.ppca = ppca;
    z.final_source_estimates = W'*ppca.marginal_source_mean;
    if ( R == 1 )
        z.final_mixing_matrix = z.ppca.A{1}*W;
    end
%=========================================================================


end