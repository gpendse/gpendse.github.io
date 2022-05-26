function z = ppca_mm(X, R, q)
%=========================================================================
%   COPYRIGHT NOTICE AND LICENSE INFO:
%
%   * Copyright (C) 2010, Gautam V. Pendse, McLean Hospital
%   
%   * Helper code for the Matlab implementation of the algorithm for estimating the PMOG model as described in the paper:
%   
%       Gautam V. Pendse, "PMOG: The projected mixture of Gaussians model with application to blind source separation", 
%       arXiv:1008.2743v1 [stat.ML, cs.AI, stat.ME], 46 pages, 9 figures, 2010. [5.5 MB] 
%
%   * This code does the Mixture of Probabilistic PCA (PPCA MM) based estimation whose results are then subjected to PMOG based BSS.
%
%   * License: Please see the file license.txt included with the code distribution
%              
%   
%   * If you use this code in your own work or alter this code, please cite the above paper. This code is distributed "as is" and no 
%   guarantee is given as to the accuracy or validity of the results produced by running this code. No responsibility will be assumed 
%   of the effects of running this code or using its results in any manner.
%
%   *   AUTHOR: GAUTAM V. PENDSE
%       DATE: 19 March 2010
%=========================================================================
%
%=========================================================================
%   PURPOSE:
%
%   * Fit a mixture of probabilistic PCA models to input data. For more details on PPCA mixture model please see the paper:
%     
%       Tipping, M. E. and C. M. Bishop (1999b). Mixtures of probabilistic principal component analysers. 
%       Neural Computation  11(2), 443?482.
%=========================================================================
%
%=========================================================================
%   MODEL:
%
%   * Given data X = (x1,x2,...,xn) where xi are p dimensional vectors. There are latent variables yi associated with each xi. yi can 
%   take one of the R values 1,2,...R. The mixture of PPCA model is given by:
%
%       P(xi | yi = k) ~ N( mu_k, A_k A_k^T + sigma_k^2 I_p)
%       [ Equivalently P(xi|yi=k, zi) ~ N(mu_k + A_k zi, sigma_k^2 I_p) where zi ~ N(0,I_q_k) ]
%
%   * The prior probabilities are given by:
%
%       p(yi = k) = pi_k
%
%   * We would like to maximize the log-likelihood of observing the data: P(X) = sum_{i = 1}^n ln(P(xi)) = sum_{i=1}^n ln[ sum_{k=1}^R pi_k P(xi|yi=k) ]
%   w.r.t the parameters mu_k, A_k, sigma_k^2 and pi_k. See NOTES for derivation of the EM algorithm for this model.
%
%   * We assume that rank A_k is q_k and that q_k is given. We assume that R, the number of PPCA components in the mixture is also given.
%=========================================================================
%
%=========================================================================
%   INPUTS:
%
%       => X = p by n matrix of n vectors in R^p
%       => R = number of PPCA components to estimate
%       => q = a vector of length R containing the ranks for each PPCA components
%=========================================================================
%   OUTPUTS:
%   
%       * save inputs  
%
%       => z.R = R;
%       => z.q = q;
%       => z.p = p;
%       => z.n = n;
%       => z.X = X;
% 
%       * save parameter estimates
%
%       => z.mu = mu;
%       => z.A = A;
%       => z.sigma2 = sigma2;
%       => z.pic = pic;
%       => z.found = found; % found = 1 means convergence and found = 2 means iteration limit reached
%     
%       * save posterior probs and likelihood, let estep = doEstep( X, mu, A, sigma2, pic );
%
%       => z.Q = estep.Q; % R by n matrix where estep.Q(k,i) = posterior prob that point i is in class k = P(yi = k | xi)
%       => z.ppca_mm_likelihood = likelihood over algorithm iterations
%=========================================================================


%=========================================================================
%               check input args
    if ( nargin ~= 3 )
        disp('Usage: z = ppca_mm(X, R, q)');
        z = [];
        return;
    end
%=========================================================================


%=========================================================================
%               check input sizes
    [p, n] = size(X); % we will assume that n >> p
 
    % make q into a col vector
    if ( size(q,2) ~= 1 )
        q = q';
    end
    
    if ( size(q,2) ~= 1 || length(q) ~= R )
        disp('q must be a vector of length R!');
        z = [];
        return;
    end
    
    for j = 1:length(q)
       if ( q(j) > p )
           disp('elements of q must be < p!');
           z = [];
           return;
       end        
    end
%=========================================================================

%=========================================================================
%               initialize PPCA components
    %o = initialize_PPCA_components( X, R, q );
    o = initialize_PPCA_components_kmeans( X, R, q );

    mu = o.mu;
    A = o.A;
    sigma2 = o.sigma2;
    pic = o.pic; 
%=========================================================================

%=========================================================================
%              start while loop

    found = 0; % convergence flag
    maxit = 1000; % maximum number of iterations
    count = 1; % counter
    
    
    while( found == 0 )
        
       % do the E-step
       estep = doEstep( X, mu, A, sigma2, pic ); 
       % get posterior probs given the params
       Q = estep.Q;
       
       % do the M-step
       mstep = doMstep( X, Q, q );         
             
       % check convergence
       found = check_convergence( mu, A, sigma2, pic, mstep, 1e-6 ); 
       
       % check number of iterations
       if ( count >= maxit )
           found = 2;
       end
 
       % update parameters
       mu = mstep.mu;
       A = mstep.A;
       sigma2 = mstep.sigma2;
       pic = mstep.pic;
       
       % compute likelihood
       likelihood(count) = compute_likelihood( X, mu, A, sigma2, pic );
       
       % plot convergence info
       if ( count == 1 )
            fh = figure('Color',[1,1,1],'Visible','on');orient(fh,'portrait');set(fh,'Units','inches','Position',[1,1,9,5.62]);
            ax = axes;
       end
       plot( ax, 1:count, likelihood, 'color','r','Marker','o','MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor','r' );
       xlabel('iteration');ylabel('likelihood');pause(0.1);
              
       % update counter
       count = count + 1;
       
    end
%=========================================================================

%=========================================================================
    %   compute posterior mean and variance for each source
    equal_dims = 1;
    for k = 1:R
       if (q(k) ~= q(1))
           equal_dims = 0;
       end
    end
    
    if ( equal_dims == 1 )
        for k = 1:R
            post_source_mean{k} = inv( A{k}'*A{k} + sigma2{k} * eye(q(k)) ) * A{k}' * ( X - repmat(mu{k},1,n) ); % q(k) by n matrix
            post_source_covar{k} = sigma2{k} * inv( A{k}'*A{k} + sigma2{k} * eye(q(k)) ); % q(k) by q(k) matrix
        end
        z.post_source_mean = post_source_mean;
        z.post_source_covar = post_source_covar;
    end
    
    % compute marginal source mean (over R classes)
    marginal_source_mean = zeros( q(1), n );
    for k = 1:R
        marginal_source_mean = marginal_source_mean + post_source_mean{k} * spdiags( Q(k,:)', 0, n, n );
    end
    z.marginal_source_mean = marginal_source_mean;
%=========================================================================

%=========================================================================
    % save inputs  
     z.R = R;
     z.q = q;
     z.p = p;
     z.n = n;
     z.X = X;

    % save parameter estimates
    z.mu = mu;
    z.A = A;
    z.sigma2 = sigma2;
    z.pic = pic;
    z.found = found; % found = 1 means convergence and found = 2 means iteration limit reached
    
    % save posterior probs and likelihood
    estep = doEstep( X, mu, A, sigma2, pic );
    
    Q = estep.Q; % R by n matrix where estep.Q(k,i) = posterior prob that point i is in class k = P(yi = k | xi)
    z.Q = Q;
    z.ppca_mm_likelihood = likelihood;
%=========================================================================        

end

function o = compute_likelihood( X, mu, A, sigma2, pic )
%=========================================================================
%               PURPOSE:
%   Compute the likelihood under the PPCA model
%=========================================================================

    % get size of X
    [p,n] = size(X);

    % get number of PPCA components
    R = length(mu);
    
    % compute pic(k) * P(xi | yi = k)
    phi = zeros(R, n);
    for k = 1:R
        
        phi(k,:) = ( pic{k} * mvnpdf( X', mu{k}', A{k}*A{k}' + sigma2{k}*eye(p) ) )'; % this will be a 1 by n vector 
    end
    
    o = sum( log( sum(phi,1) ) );
    
end


function o = initialize_PPCA_components( X, R, q )
%=========================================================================
%               PURPOSE:
%   Responsible for initializing PPCA components
%=========================================================================

    [p,n] = size(X);

    for k = 1:R
       mu{k} = randn( p, 1 ); % random mean vector
       A{k} = randn( p, q(k) ); % random matrix
       sigma2{k} = 10*mean(std(X,1))^2; % unit noise variance
       pic{k} = 1/R; % equal class proportions
    end

    o.mu = mu;
    o.A = A;
    o.sigma2 = sigma2;
    o.pic = pic;

end

function o = initialize_PPCA_components_kmeans( X, R, q )
%=========================================================================
%               PURPOSE:
%   Responsible for initializing PPCA components
%=========================================================================

    [p,n] = size(X);

    [IDX, C] = kmeans(X', R, 'Replicates', 10, 'EmptyAction', 'singleton'); % X' is n by p, C should be R by p
    
    for k = 1:R
       mu{k} = C(k,:)'; % mean vector from kmeans
       A{k} = randn( p, q(k) ); % random matrix
       sigma2{k} = 10*mean(std(X,1))^2; % unit noise variance
       pic{k} = 1/R; % equal class proportions
    end

    o.mu = mu;
    o.A = A;
    o.sigma2 = sigma2;
    o.pic = pic;

end

function o = doEstep( X, mu, A, sigma2, pic )
%=========================================================================
%               PURPOSE:
%   Perform the E-step of estimating a PPCA mixture
%   X = p by n matrix of n vectors in R^p
%   mu = length R cell array containing the mean vectors of each PPCA
%   A = length R cell array containing the A matrix for each PPCA
%   sigma2 = length R cell array containing the noise variance of each PPCA
%   pic = length R cell array containing the class fraction of each PPCA
%=========================================================================

    % get number of PPC components
    R = length(pic);

    % get size of X
    [p,n] = size(X);
    
    % compute the numerator in E-step for all i
    phi = zeros(R, n);
    for k = 1:R
        phi(k,:) = ( pic{k} * mvnpdf( X', mu{k}', A{k}*A{k}' + sigma2{k}*eye(p) ) )'; % this will be a 1 by n vector 
    end
    
    % compute the denominator in E-step for all i
    sum_phi = sum(phi,1); % 1 by n
        
    % compute the posterior probability matrix Q
    Q = zeros(R,n);
    
    Q = phi ./ repmat( sum_phi, R, 1 ); % R by n
    
    % replace NaN's in Q by 1/R
    Q( :, find( sum(isnan(Q),1) == R ) ) = 1/R;
    
    o.Q = Q; % R by n matrix Q(yi = k) is given by Q(k,i)
    
end

function o = doMstep( X, Q, q )
%=========================================================================
%               PURPOSE:
%   Perform the M-step of estimating a PPCA mixture
%   X = p by n matrix of n vectors in R^p
%   Q = R by n posterior probability matrix from E-step
%   q = length R vector containing the rank of each A{k}
%=========================================================================

    % get size X
    [p,n] = size(X);
    
    % get size of Q
    [R,n1] = size(Q);
    
    % check Q and X sizes
    if ( n1 ~= n )
        disp('Q must have the same number of columns as X!');
        o = [];
        return;
    end
    
    % check size of q
    if ( length(q) ~= R )
        disp('q must be of length R!');
        o = [];
        return;
    end
    
    
    sumQ = sum(sum(Q));
    % get pic{k}
    for k = 1:R
       pic{k} = sum(Q(k,:))/sumQ; 
    end
    
    
    % get mu{k}
    for k = 1:R
       mu{k} = (X * Q(k,:)')/sum(Q(k,:)); % p by n times n by 1 = p by 1        
    end

    % get modified sample covariance
    for k = 1:R
       
        X_muk = X - repmat(mu{k},1,n); % p by n matrix ith col is xi - mu{k}
        
        temp = sqrt( repmat( Q(k,:), p, 1 ) ) .* X_muk; % compute a matrix whose ith col is sqrt(Q(yi = k)) (xi - mu{k})
        
        S{k} = (temp * temp') / sum(Q(k,:));
        
    end
   
    % get sigma2 and A
    for k = 1:R
       
        %[Ug,Sg,Vg] = svds( S{k}, q(k) );
        [Ug,Sg,Vg] = svd( S{k}, 0 );
        Ug = Ug(:,1:q(k));
        Sg = Sg(1:q(k),1:q(k));
        
        if ( p == q(k) )
            sigma2{k} = 0;
        else
            sigma2{k} = ( trace(S{k}) - trace(Sg) )/(p - q(k));
        end
        A{k} = Ug * sqrt(Sg - sigma2{k} * eye(q(k)));       
    
    end
    
    % make sure variance is sufficiently positive
%     for k = 1:R
%        if ( sigma2{k} <= 1e-3 )
%            sigma2{k} = 1e-3;
%        end
%     end
    
    % save outputs
    o.mu = mu;
    o.A = A;
    o.sigma2 = sigma2;
    o.pic = pic;

end

function found = check_convergence( mu, A, sigma2, pic, mstep, tol )
%=========================================================================
%               PURPOSE:
%   check convergence of EM iteration
%   mu = length R cell array containing the mean vectors of each PPCA
%   A = length R cell array containing the A matrix for each PPCA
%   sigma2 = length R cell array containing the noise variance of each PPCA
%   pic = length R cell array containing the class fraction of each PPCA
%   mstep = structure returned by M-step
%   tol = convergence tolerance
%=========================================================================

    R = length(mu);
    
    found = 1;
    
    % check mu
    for k = 1:R        
        if ( max(abs( mu{k} - mstep.mu{k} )) > tol )
            found = 0;
        end       
    end

    % check A
    for k = 1:R
       
        if ( max(max( abs( A{k}*A{k}' - mstep.A{k}*mstep.A{k}' ) )) > tol )
            found = 0;
        end
        
    end
    
    % check sigma2
    for k = 1:R
       
        if ( max(abs( sigma2{k} - mstep.sigma2{k} ) ) > tol )
            found = 0;
        end
        
    end
    
    
    % check pic    
    for k = 1:R
    
        if ( max( abs( pic{k} - mstep.pic{k} ) ) > tol )
            found = 0;
        end
        
    end
    
    
end


