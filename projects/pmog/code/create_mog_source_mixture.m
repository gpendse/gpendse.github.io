function z = create_mog_source_mixture( nsources, rowsA, samples_per_source )
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
%   * This code creates a MOG source mixture:
%
%   * AUTHOR: GAUTAM V. PENDSE
%     PURPOSE: Create a MOG source mixture
%=========================================================================
%
%=========================================================================
%   INPUTS:
%
%       =>           nsources = number of MOG sources in the mixture
%       =>              rowsA = number of rows in the mixing matrix
%       => samples_per_source = number of sample points per source
%=========================================================================
%
%=========================================================================
% 	TECHNIQUE:
%
%       =>  nsources MOG sources are generated as follows:
%           (a) pic = rand(5, 1); pic = pic/sum(pic);
%           (b) mu = unifrnd(-10,10, 5, 1);
%           (c) sigma2 = unifrnd(1,5, 5, 1);
%
%       => Let S be a matrix such that rows of S represent the sources.
%           (1) Demean each row of S
%               S = S - repmat( mean(S,2), 1, size(S,2) )
%           (2) Make S unit co-variance
%               Let [U,Sig,V] = svd( S*S'/samples_per_source, 0 );
%               S = diag((diag(Sig))^(-0.5))*U'*S
%
%       => Generate a rowsA by nsources random matrix A and create 
%           X = A*S
%
%       => Return, A, S and X
% 
%=========================================================================

%=========================================================================
%               check input args
        if ( nargin ~= 3 )
            disp('Usage: z = create_mog_source_mixture( nsources, rowsA, samples_per_source )');
            z = [];
            return;
        end
%=========================================================================

%=========================================================================
%               First generate sources
        S = zeros(nsources, samples_per_source);
        
        for k = 1:nsources
            
            pic = rand(5,1);
            pic = pic/sum(pic);
            
            mu = unifrnd(-10,10,5,1);
            sigma2 = unifrnd(1,5,5,1);
            
            mogs = create_mog_source(pic, mu, sigma2, samples_per_source);
            
            S(k, :) = mogs.samples';            
        end
%=========================================================================

%=========================================================================
%               Make sources zero mean and unit covariance
        S = S - repmat(mean(S,2),1,samples_per_source);
        
        [U,Sig,V] = svd( S*S'/samples_per_source, 0 );
        
        S = diag((diag(Sig)).^(-0.5))*U'*S;
%=========================================================================

%=========================================================================
%              Generate mixing matrix
        A = rand( rowsA, nsources );
%=========================================================================

%=========================================================================
%              Generate mixture
        X = A*S; % if passing to ppca_mm_ica please add sqrt(sigma2)*randn( size(X,1), size(X,2) ) Gaussian noise
%=========================================================================

%=========================================================================
%              save outputs

        z.X = X;
        z.A = A;
        z.S = S;
%=========================================================================
end