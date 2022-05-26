function o = projected_mog_ica(Z, R, G, tol, wuser)
%=========================================================================
%   COPYRIGHT NOTICE AND LICENSE INFO:
%
%   * Copyright (C) 2010, Gautam V. Pendse, McLean Hospital
%   
%   * Matlab implementation of the algorithm for estimating the PMOG model as described in the paper:
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
%     DATE: 26th March 2010
%=========================================================================
%
%=========================================================================
%   PURPOSE:
%	
%   * Projected MOG (PMOG) estimation algorithm for ICA
%
%   * At the last stage in ICA, an ICA source element is estimated as u_i = w^T z_i  where z_i is a q by 1 vector and w is a q by 1 vector. 
%   The n elements of u_i for i = 1,2,...n are supposed to have an empirical density that is as non-Gaussian as possible.
%
%   * In this work, we propose to model u_i as samples from an R component MOG density. An MoG density is a flexible non-Gaussian density. 
%   We estimate the projected MOG (PMOG) density using an EM algorithm derived in the paper above. We also impose the following constraints on w:
%   
%       (1) w^T w = 1 (unit norm) and 
%       (2) G^T w = 0 where G is a given q by L matrix with L < q
%=========================================================================
%
%=========================================================================
%   INPUTS:
%   
%    =>     Z = q by n matrix of vectors in R^q with q < n
%    =>     R = number of components in the projected MOG
%    =>     G = q by L matrix with L < q (to be used for imposing G^T w = 0 constraint) 
%    =>   tol = relative tolerance for stopping iterations (default = 1e-5)
%    => wuser = initial value for vector w (optional, default = random)
%
%=========================================================================
%   OUTPUTS:
%
%       * save inputs  
%    => o.Z = Z;
%    => o.R = R;
%    => o.G = G;
%        
%       * save parameter estimates
%    => o.w = w (computed projection vector)
%
%       * mixture model parameters in cell arrays
%    => o.pic = pic;
%    => o.mu = mu;
%    => o.sigma2 = sigma2;
%    => o.found = found; % found = 1 means convergence and found = 2 means iteration limit reached
%     
%       * save posterior probs and likelihood, let estep = doEstep( Z, w, pic, mu, sigma2 );
%    => o.alpha = estep.alpha; % R by n matrix where estep.alpha(k,i) = posterior prob that point i is in class k = P(yi = k | w'*zi)     
%    => z.projected_mog_likelihood = likelihood over algorithm iterations
%=========================================================================

%=========================================================================
%               check input args
    if ( nargin < 3 )
        disp('Usage: o = projected_mog_ica(Z, R, G) or o = projected_mog_ica(Z, R, G, tol) or o = projected_mog_ica(Z, R, G, tol, wuser)');
        help('projected_mog_ica.m');
        o = [];
        return;
    end
%=========================================================================

%=========================================================================
%       set tolerance
    if ( nargin == 3 )
        tol = 1e-5;
        wuser = [];
    end
    
%       set default winit
    if ( nargin == 4 )
        wuser = [];
    end
    
    if ( nargin == 5 )
        
       % make winit into a column vector
       if ( size(wuser,2) ~= 1 )
           wuser = wuser';
       end
       
       % make sure winit is a vector
       if ( size(wuser,2) ~= 1 )
           disp('wuser must be a vector!')
           o = [];
           return;
       end
       
       % make sure winit is of the right size
       if ( size(wuser,1) ~= size(Z,1) )
           disp('wuser must be of the same length as the number of rows in Z!');
           o = [];
           return;
       end
       
    end
%=========================================================================

%=========================================================================
%               validate input
    [q,n] = size(Z);
    if ( q >= n )
        disp('Z should be q by n with q < n!');
        o = [];
        return;
    end
    
    if ( isempty(G) == 1 )
    else
        [q1,L] = size(G);
        if ( L >= q1 || q1 ~= q )
            disp('If Z is q by n then G should be a q by L matrix with L < q');
            o = [];
            return;
        end
    end
%=========================================================================

%=========================================================================
%              define priors and initialize parameters
   
    % dirichlet prior on class fractions
    beta = (1+1e-3)*ones(R,1);

    % define inverse gamma prior parameters for each sigma2{k}
    sigma2_a = zeros(R,1);
    sigma2_b = zeros(R,1);
    for k = 1:R
       sigma2_a(k) = 2;%0.1; %2;
       sigma2_b(k) = 1;%1e3; %1;
    end
    
    % initialize parameters
    init = initialize_parameters_kmeans2( Z, R, G, beta, sigma2_a, sigma2_b, wuser  );
    w = init.w;
    pic = init.pic;
    mu = init.mu;
    sigma2 = init.sigma2;

    
    % has the user provided initialization?
    if ( isempty( wuser ) == 0 )
       w = wuser; 
    end
    
    % select w using select_good_point
    %w = select_good_seed_point(Z, G, pic, mu, sigma2, beta, sigma2_a, sigma2_b);       
%=========================================================================

%=========================================================================
%             begin while loop
   
    % convergence flag
    found = 0;
    
    % iteration counter
    count = 1;
    
    % max iterations
    maxit = 1000;
    
    while( found == 0 )
        
        % compute likelihood
        likelihood(count) = compute_likelihood( Z, w, pic, mu, sigma2, beta, sigma2_a, sigma2_b );
               
        % do E-step
        estep = doEstep( Z, w, pic, mu, sigma2 );
        
        % do M-step
        winit = w;
        mstep = doMstep( Z, estep.alpha, G, winit, pic, mu, sigma2, beta, sigma2_a, sigma2_b );
        
        while ( mstep.found ~= 1 || compute_likelihood_Q( Z, mstep.w, mstep.pic, mstep.mu, mstep.sigma2, beta, sigma2_a, sigma2_b ) < compute_likelihood_Q( Z, w, pic, mu, sigma2, beta, sigma2_a, sigma2_b ) )
        %while ( mstep.found ~= 1 || compute_likelihood( Z, mstep.w, mstep.pic, mstep.mu, mstep.sigma2, beta, sigma2_a, sigma2_b ) < likelihood(count) )
            %winit = rand(q,1);
            
            % select winit using select_good_seed_point
            winit = select_good_seed_point(Z, G, pic, mu, sigma2, beta, sigma2_a, sigma2_b);       
            
            if ( isempty(G) == 1 )
                winit = winit;
            else
                winit = winit - G*inv(G'*G)*G'*winit;
            end
            winit = winit/norm(winit);

            mstep = doMstep( Z, estep.alpha, G, winit, pic, mu, sigma2, beta, sigma2_a, sigma2_b );
            
            mstep
        end
        
        % check convergence
        %found = check_convergence( mstep.w, mstep.pic, mstep.mu, mstep.sigma2, w, pic, mu, sigma2, 1e-3 );

        % check number of iterations
        if ( count >= maxit )
            found = 2;
        end
        
        % update parameters
        w = mstep.w;
        pic = mstep.pic;
        mu = mstep.mu;
        sigma2 = mstep.sigma2;
        
       
       % check convergence based on likelihood
       if ( count > 1 )
          
           disp(['Likelihood error: ',num2str( abs( likelihood(count) - likelihood(count - 1) ) )]);
           pause(1);
           
           %if ( abs( likelihood(count) - likelihood(count - 1) ) < tol )
           if ( abs( likelihood(count) - likelihood(count - 1) ) < tol * abs(mean(likelihood(1:count))) )
               % use relative tolerance

               found = 1;
           end
           
       end
       
       % plot convergence info
       if ( count == 1 )
            fh = figure('Color',[1,1,1],'Visible','on');orient(fh,'portrait');set(fh,'Units','inches','Position',[1,1,9,5.62]);
            %ax = axes;
            ax1 = subplot(2,1,1);
            ax2 = subplot(2,1,2);
       end
       plot( ax1, 1:count, likelihood, 'color','r','Marker','o','MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor','r' );
       xlabel(ax1, 'iteration');ylabel(ax1,'likelihood');pause(0.1);
              
       %hist( ax2, w'*Z, 100 );
       plot_current_mog( ax2, Z, w, pic, mu, sigma2 );
       xlabel(ax2,'projected points');ylabel(ax2,'density');
       %uTest = w'*Z;
       %title( ax2,['Unit norm negentropy: ',num2str( compute_negentropy_for_vector( uTest/norm(uTest) ) )] );
       pause(0.1);
       % update counter
       count = count + 1;
        
    end    
%=========================================================================


    % save inputs  
    o.Z = Z;
    o.R = R;
    o.G = G;
    
  
    % save parameter estimates
    o.w = w;
    o.pic = pic;
    o.mu = mu;
    o.sigma2 = sigma2;
    o.found = found; % found = 1 means convergence and found = 2 means iteration limit reached
    
    % save posterior probs and likelihood
    estep = doEstep( Z, w, pic, mu, sigma2 );
    o.alpha = estep.alpha; % R by n matrix where estep.alpha(k,i) = posterior prob that point i is in class k = P(yi = k | w'*zi)
    
    o.projected_mog_likelihood = likelihood;

end


function w = select_good_seed_point(Z, G, pic, mu, sigma2, beta, sigma2_a, sigma2_b)
%=========================================================================
%               AUTHOR: GAUTAM V. PENDSE
%               DATE: 16th April 2010
%=========================================================================
%
%=========================================================================
%               PURPOSE:
%   Select "good" vector w given other parameters such that w is near a
%   maximum of the likelihood
%=========================================================================
%
%=========================================================================
%               INPUTS:
%    Z = q by n matrix of vectors in R^q with q < n
%    G = q by L matrix with L < q (to be used for imposing G^T w = 0 constraint) 
%   pic = cell array of class fractions for the R classes
%   mu = cell array of class means for the R classes
%   sigma2 = cell array of variances for the R classes
%   beta = R by 1 vector of prior parameters for the class fractions (user specified)
%   sigma2_a = R by 1 vector of the "a" parameter for the inverse gamma prior on sigma2{k}
%   sigma2_b = R by 1 vector of the "b" parameter for the inverse gamma prior on sigma2{k}
%=========================================================================
%
%=========================================================================
%               OUTPUTS:
%    w = q by 1 vector, a "good seed point"
%=========================================================================


%==============================================================
            % winit selection using find_good_seed_points
            param.Z = Z;
            param.pic = pic;
            param.mu = mu;
            param.sigma2 = sigma2;
            param.beta = beta;
            param.sigma2_a = sigma2_a;
            param.sigma2_b = sigma2_b;
            
            % get q
            q = size(Z,1);
            
            if ( isempty(G) == 1 )
                FGS = find_good_seed_points( 'max', make_compute_likelihood_for_w(), param, 1, 1000, eye(q), 1);
            else
                FGS = find_good_seed_points( 'max', make_compute_likelihood_for_w(), param, 1, 1000, compute_orthogonal_complement(G), 1);
            end
            
            w = FGS(:,1);            
    %==============================================================

end

function z = plot_current_mog( ax, Z, w, pic, mu, sigma2 )
%=========================================================================
%               AUTHOR: GAUTAM V. PENDSE
%               DATE: 26th March 2010
%=========================================================================
%
%=========================================================================
%               PURPOSE:
%   Plot current MOG in the provided axes handle
%=========================================================================
%
%=========================================================================
%               INPUTS:
%   ax = axis handle into which to plot data
%    Z = q by n matrix of vectors in R^q with q < n
%    w = random projection vector such that w'*w = 1 and G'*w = 0
%    pic = cell array of class fractions for the R classes
%    mu = cell array of class means for the R classes
%    sigma2 = cell array of variances for the R classes
%=========================================================================
%
%=========================================================================
%               OUTPUTS:
%   z = []
%=========================================================================
    z = [];

    % get histogram of projected points
    [N,X] = hist(ax, w'*Z , 200);
    DX = X(2) - X(1);
    
    % plot observed density
    plot(ax, X,N/sum(N)/DX, 'b');
    hold on;
    E = X'; % each row of E contains an evaluation vector, E is of size m by 1
    
    % get number of comps
    R = length(pic);
    
    % compute phi_K
    for k = 1:R
        phi_X(k, :) = pic{k} * normpdf( E, mu{k}, sqrt(sigma2{k}) )';
    end
    % plot MOG density
    plot(ax, X,sum(phi_X,1),'r');hold on;

    % plot individual MOG components
    color = {'g','c','m','y','k'};
    string = {'Data','MOG fit'};
    for j = 1:R
        string{length(string) + 1} = ['Dist: ',num2str(j)];
        plot(ax, X,phi_X(j,:),color{max(mod(j,length(color)),1)});
    end

    legend(ax, string,'Location','Best');
    hold off;

end

function z = compute_likelihood_Q( Z, w, pic, mu, sigma2, beta, sigma2_a, sigma2_b )
%=========================================================================
%               AUTHOR: GAUTAM V. PENDSE
%               DATE: 26th March 2010
%=========================================================================
%
%=========================================================================
%               PURPOSE:
%   Compute lower bound on likelihood for a projected MOG model 
%   This is the Q function which is a lower bound on H_1 in the paper
%=========================================================================
%
%=========================================================================
%               INPUTS:
%    Z = q by n matrix of vectors in R^q with q < n
%    w = random projection vector such that w'*w = 1 and G'*w = 0
%    pic = cell array of class fractions for the R classes
%    mu = cell array of class means for the R classes
%    sigma2 = cell array of variances for the R classes
%    beta = R by 1 vector of prior parameters for the class fractions (user specified)
%    sigma2_a = R by 1 vector of the "a" parameter for the inverse gamma prior on sigma2{k}
%    sigma2_b = R by 1 vector of the "b" parameter for the inverse gamma prior on sigma2{k}
%=========================================================================
%
%=========================================================================
%               OUTPUTS:
%   z = log-likelihood of a projected MOG model
%=========================================================================

    % get size of Z
    [q,n] = size(Z);
    
    % get number of components in MOG
    R = length(pic);
    
    % get projections using w
    u = w'*Z; % 1 by n vector
    
    % get P(yi = k) * P(ui | yi = k) = pic * P(ui | yi = k)
    phi = zeros(R,n);
    for k = 1:R
       phi(k,:) = pic{k} * normpdf( u, mu{k}, sqrt(sigma2{k}) ); 
    end

    alpha = phi ./ repmat( sum(phi,1), R, 1 ); % R by n
    
    z = sum( sum( alpha .* log( phi ), 1 ) ) - sum( sum( alpha .* log( alpha ), 1 ) );
    
    %z = sum( log( sum(phi,1) ) );

    % get contribution of first prior term (dirichlet prior on pic{k})
    prior_term1 = 0;
    for k = 1:R
        prior_term1 = prior_term1 + (beta(k) - 1) * log( pic{k} );
    end

    % get contribution of the second prior term (inverse gamma prior on sigma2{k})
    prior_term2 = 0;
    for k = 1:R
       prior_term2 = prior_term2 - ( sigma2_a(k) + 1 ) * log(sigma2{k}) - (1 / (sigma2_b(k) * sigma2{k}) ); 
    end
    
    z = z + prior_term1 + prior_term2;
    
end

function z = compute_likelihood( Z, w, pic, mu, sigma2, beta, sigma2_a, sigma2_b )
%=========================================================================
%               AUTHOR: GAUTAM V. PENDSE
%               DATE: 26th March 2010
%=========================================================================
%
%=========================================================================
%               PURPOSE:
%   Compute likelihood for a projected MOG model
%=========================================================================
%
%=========================================================================
%               INPUTS:
%    Z = q by n matrix of vectors in R^q with q < n
%    w = random projection vector such that w'*w = 1 and G'*w = 0
%    pic = cell array of class fractions for the R classes
%    mu = cell array of class means for the R classes
%    sigma2 = cell array of variances for the R classes
%    beta = R by 1 vector of prior parameters for the class fractions (user specified)
%    sigma2_a = R by 1 vector of the "a" parameter for the inverse gamma prior on sigma2{k}
%    sigma2_b = R by 1 vector of the "b" parameter for the inverse gamma prior on sigma2{k}
%=========================================================================
%
%=========================================================================
%               OUTPUTS:
%   z = log-likelihood of a projected MOG model
%=========================================================================

    % get size of Z
    [q,n] = size(Z);
    
    % get number of components in MOG
    R = length(pic);
    
    % get projections using w
    u = w'*Z; % 1 by n vector
    
    % get P(yi = k) * P(ui | yi = k) = pic * P(ui | yi = k)
    phi = zeros(R,n);
    for k = 1:R
       phi(k,:) = pic{k} * normpdf( u, mu{k}, sqrt(sigma2{k}) ); 
    end

    z = sum( log( sum(phi,1) ) );

    % get contribution of first prior term (dirichlet prior on pic{k})
    prior_term1 = 0;
    for k = 1:R
        prior_term1 = prior_term1 + (beta(k) - 1) * log( pic{k} );
    end

    % get contribution of the second prior term (inverse gamma prior on sigma2{k})
    prior_term2 = 0;
    for k = 1:R
       prior_term2 = prior_term2 - ( sigma2_a(k) + 1 ) * log(sigma2{k}) - (1 / (sigma2_b(k) * sigma2{k}) ); 
    end
    
    z = z + prior_term1 + prior_term2;
    
end

function z = make_compute_likelihood_for_w()

    z = @f;
    
    function ret = f( w, param )
        
        ret = compute_likelihood_for_w( w, param );
        
    end

end

function z = compute_likelihood_for_w( w, param )
%=========================================================================
%               AUTHOR: GAUTAM V. PENDSE
%               DATE: 26th March 2010
%=========================================================================
%
%=========================================================================
%               PURPOSE:
%   Compute likelihood for a projected MOG model
%=========================================================================
%
%=========================================================================
%               INPUTS:
%    w = random projection vector such that w'*w = 1 and G'*w = 0
%    param.Z = q by n matrix of vectors in R^q with q < n
%    param.pic = cell array of class fractions for the R classes
%    param.mu = cell array of class means for the R classes
%    param.sigma2 = cell array of variances for the R classes
%    param.beta = R by 1 vector of prior parameters for the class fractions (user specified)
%    param.sigma2_a = R by 1 vector of the "a" parameter for the inverse gamma prior on sigma2{k}
%    param.sigma2_b = R by 1 vector of the "b" parameter for the inverse gamma prior on sigma2{k}
%=========================================================================
%
%=========================================================================
%               OUTPUTS:
%   z = log-likelihood of a projected MOG model
%=========================================================================

    % extract values from param struct
    Z = param.Z;
    pic = param.pic;
    mu = param.mu;
    sigma2 = param.sigma2;
    beta = param.beta;
    sigma2_a = param.sigma2_a;
    sigma2_b = param.sigma2_b;

    z = compute_likelihood( Z, w, pic, mu, sigma2, beta, sigma2_a, sigma2_b );

end

function estep = doEstep( Z, w, pic, mu, sigma2 )
%=========================================================================
%               AUTHOR: GAUTAM V. PENDSE
%               DATE: 26th March 2010
%=========================================================================
%
%=========================================================================
%               PURPOSE:
%   Do E-step for projected MOG
%=========================================================================
%
%=========================================================================
%               INPUTS:
%    Z = q by n matrix of vectors in R^q with q < n
%    w = random projection vector such that w'*w = 1 and G'*w = 0
%    pic = cell array of class fractions for the R classes
%    mu = cell array of class means for the R classes
%    sigma2 = cell array of variances for the R classes
%=========================================================================
%
%=========================================================================
%               OUTPUTS:
%    estep.alpha = R by n matrix where alpha(k,i) = P(yi = k | ui)
%=========================================================================

    % get size of Z
    [q,n] = size(Z);
    
    % get number of components in MOG
    R = length(pic);
    
    % get projections using w
    u = w'*Z; % 1 by n vector
    
    % get P(yi = k) * P(ui | yi = k) = pic * P(ui | yi = k)
    phi = zeros(R,n);
    for k = 1:R
       phi(k,:) = pic{k} * normpdf( u, mu{k}, sqrt(sigma2{k}) ); 
    end

    % compute alpha(k,i) = P(yi = k | ui)
    alpha = phi ./ repmat( sum(phi,1), R, 1 ); % R by n
    
    % replace NaN's in alpha by 1/R
    alpha( :, find( sum(isnan(alpha),1) == R ) ) = 1/R;
    
    % save output
    estep.alpha = alpha;
    
end

function mstep = doMstep( Z, alpha, G, winit, pic, mu, sigma2, beta, sigma2_a, sigma2_b )
%=========================================================================
%               AUTHOR: GAUTAM V. PENDSE
%               DATE: 26th March 2010
%=========================================================================
%
%=========================================================================
%               PURPOSE:
%   Do M-step for projected MOG
%=========================================================================
%
%=========================================================================
%               INPUTS:
%    Z = q by n matrix of vectors in R^q with q < n
%    alpha = R by n matrix where alpha(k,i) = P(yi = k | ui)
%    G = q by L matrix with L < q (to be used for imposing G^T w = 0 constraint) 
%   winit = q by 1 vector to start the iterative M-step
%   pic = cell array of class fractions for the R classes
%   mu = cell array of class means for the R classes
%   sigma2 = cell array of variances for the R classes
%   beta = R by 1 vector of prior parameters for the class fractions (user specified)
%   sigma2_a = R by 1 vector of the "a" parameter for the inverse gamma prior on sigma2{k}
%   sigma2_b = R by 1 vector of the "b" parameter for the inverse gamma prior on sigma2{k}
%=========================================================================
%
%=========================================================================
%               OUTPUTS:
%    mstep.w = projection vector such that w'*w = 1 and G'*w = 0
%    mstep.pic = cell array of class fractions for the R classes
%    mstep.mu = cell array of class means for the R classes
%    mstep.sigma2 = cell array of variances for the R classes
%=========================================================================


    % get size of Z
    [q,n] = size(Z);
    
    R = size(alpha,1); % alpha is R by n
    
    %=========================================================================
    %   start iterative M-step
        
        % convergence flag
        found = 0;
        
        % counter
        count = 1;
        
        % maximum iterations
        maxit = 1000;
        
        % initialize w
        w = winit;
        while( found == 0 )
            
           % compute likelihood
           likelihood(count) = compute_likelihood( Z, w, pic, mu, sigma2, beta, sigma2_a, sigma2_b );

           % check if M-step failed
%            if ( count > 1 )
%               % if likelihood decreases after computing w mark the M-step as failed
%               if ( likelihood(count) < likelihood(count - 1) )
%                   disp('M-step failed!');
%                   keyboard;
%                   found = 2;
%               end
%            end           
%            
           % check number of iterations
           if ( count >= maxit )
               found = 2;
           end
           
           % check convergence based on likelihood
           if ( count > 1 )
                disp(['likelihood error: ',num2str( abs( likelihood(count) - likelihood(count - 1) ) )]);
                if ( abs( likelihood(count) - likelihood(count - 1) ) < 1e-3 && found ~= 2 )
                    found = 1;
                    %keyboard;
                end                
           end
           
           if ( found == 0 )
           
               % compute pic
               sum_sum_alpha = sum(sum(alpha));
               sum_beta = sum(beta);
               for k = 1:R
                  pic{k} = ( sum(alpha(k,:)) + (beta(k) - 1) ) / ( sum_sum_alpha + sum_beta - R );
                  
                  if ( pic{k} < 0 )
                      keyboard;
                  end
               end

               % compute projections
               u = w'*Z; % 1 by n vector

               % compute mu
               for k = 1:R
                   mu{k} = (alpha(k,:)*u')/sum(alpha(k,:));
               end

%                for k = 1:R
%                   if ( isnan(mu{k}) == 1 )
%                       mu{k} = 0;
%                   end
%                end

               % compute sigma2
               for k = 1:R
                  sigma2{k} = ( sum( alpha(k,:) * ((u' - mu{k}).^2) ) + 2/sigma2_b(k) ) / ( sum(alpha(k,:)) + 2*(sigma2_a(k) + 1) );
               end

%                % change 0 variance to eps variance
%                for k = 1:R
%                    if ( sigma2{k} == 0 )
%                       sigma2{k} = eps; 
%                    end
%                end
%                for k = 1:R
%                   if ( isnan(sigma2{k}) == 1 )
%                       sigma2{k} = 1;
%                   end
%                end

               % Given w, we have already computed closed form solns for pic,
               % mu and sigma2. Now we compute w by doing a newton iteration
               % fixing all other variables
               % compute w
               % (1) compute b
               b = zeros(q,1);
               for k = 1:R
                  b = b + (mu{k}/sigma2{k}) * ( Z * alpha(k,:)' );
               end
               % (2) compute A
               A = zeros(q,q);
               for k = 1:R
                  A = A + (1/sigma2{k}) * ( Z * spdiags( alpha(k,:)',0,n,n ) * Z' );
               end
               % (3) compute orthogonal projector to G
               if ( isempty(G) == 1 )
                   PG = eye(q);
               else
                   PG = eye(q) - G*inv(G'*G)*G';
               end

               try
                   sfw = solve_for_w( PG, b, A );
               catch
                   keyboard; 
               end
               
               w = sfw.w;
           
           end
           
           
           disp(['current f = ',mat2str( w_equation( PG, b, A, w )' ,2)]);           
           w
           pic
           mu
           sigma2
           pause(0.5);

                      
           %likelihood(count) = compute_likelihood( Z, w, pic, mu, sigma2 );
           
           
%            T = -A + n * eye(q); 
%            disp(['max(eig(T)) = ',num2str(max(eig(T)))]);
%            pause(1);

           
           if ( isnan(w) == 1 )
               keyboard;
           end
           
%            if ( count > 1 )
%                 % check convergence
%                 %found = check_convergence( w, pic, mu, sigma2, w_old, pic_old, mu_old, sigma2_old, 1e-3 );
%                 %keyboard;
%            end
       
       
           
           
%            if ( rem(count,100) == 0 )
%                %keyboard;
%            end
           
%            % save current estimates to check convergence
%            w_old = w;
%            pic_old = pic;
%            mu_old = mu;
%            sigma2_old = sigma2;

           
           
           
           % update counter
           count = count + 1;
           
        end        
    %=========================================================================    
 
    % save outputs
    mstep.w = w;
    mstep.pic = pic;
    mstep.mu = mu;
    mstep.sigma2 = sigma2;
    mstep.found = found;
    
end

function o = solve_for_w( PG, b, A )
%=========================================================================
%               AUTHOR: GAUTAM V. PENDSE
%               DATE: 29th March 2010
%=========================================================================
%
%=========================================================================
%               PURPOSE:
%   Let f(w) = PG*(b-A*w) - (w'*b - (w'*A*w))*w
%   For a given vector w, the output will also be a q by 1 vector. We would
%   like this vector to be zero. Here we use a newton iteration to get w such 
%   that f(w) = 0.
%=========================================================================
%
%=========================================================================
%               INPUTS:
%    PG, b and A are quantities independent of w
%=========================================================================

       % convergence flag
       found_w = 0;
       
       % initialize w
       
       w = pinv(A)*b;
       w = PG*w;
       w = w/norm(w);
       
       while( found_w == 0 )
 
           
          % compute  the Jacobian at current value of w
          J = jacobian_anal_w_equation( PG, b, A , w ); 
              
          % compute current vector function
          f = w_equation( PG, b, A, w );
          
          if ( max(abs(f)) < 1e-4 )
              found_w = 1;
          end
          
          if ( found_w == 0 )
            % compute new w
            w_new = w - inv(J)*f;
            w = w_new;
          end          
          
%           % impose constraints
%           w_new = PG*w_new;
%           
%           w_new = w_new/norm(w_new)
          
          %pause(0.1);
          
%           if ( max(abs(w_new - w)) < 1e-4 || max(abs(w_new + w)) < 1e-4 )
%               found_w = 1;
%           end


       end

       o.w = w;
end

function f = w_equation( PG, b, A , w )
%=========================================================================
%               AUTHOR: GAUTAM V. PENDSE
%               DATE: 29th March 2010
%=========================================================================
%
%=========================================================================
%               PURPOSE:
%   Let f(w) = PG*(b-A*w) - (w'*b - (w'*A*w))*w
%   For a given vector w, the output will also be a q by 1 vector. We would
%   like this vector to be zero. In fact, we will use this function in the
%   code solve_for_w to perform a newton iteration to get w such that f(w)
%   = 0.
%=========================================================================
%
%=========================================================================
%               INPUTS:
%    PG, b and A are quantities independent of w
%    w is the vector at which we wish to evaluate f(w)
%=========================================================================

    f = PG*(b-A*w) - (w'*b - (w'*A*w))*w;


end

function J = jacobian_w_equation( PG, b, A , w )
%=========================================================================
%               AUTHOR: GAUTAM V. PENDSE
%               DATE: 29th March 2010
%=========================================================================
%
%=========================================================================
%               PURPOSE:
%   Let f(w) = PG*(b-A*w) - (w'*b - (w'*A*w))*w
%   For a given vector w, the output will also be a q by 1 vector. We would
%   like this vector to be zero. In fact, we will use this function in the
%   code solve_for_w to perform a newton iteration to get w such that f(w)
%   = 0. This code will return the Jacobian of f(w) at a given value of w.
%=========================================================================
%
%=========================================================================
%               INPUTS:
%    PG, b and A are quantities independent of w
%    w is the vector at which we wish to evaluate f(w)
%=========================================================================

    % define delta for numerical gradients
    delta = eps^(1/3);
    
    % get length of w
    q = length(w);
    
    % initialize Jacobian
    J = zeros(q,q);
    
    for k = 1:q
       % define w2
       w2 = w;
       w2(k) = w2(k) + delta;
       
       % define w1
       w1 = w;
       w1(k) = w1(k) - delta;
       
       J(:,k) = ( w_equation( PG, b, A, w2 ) - w_equation( PG, b, A, w1 ) )/2/delta;
       
    end

    %f = PG*(b-A*w) - (w'*b - (w'*A*w))*w;    

end

function J = jacobian_anal_w_equation( PG, b, A , w )
%=========================================================================
%               AUTHOR: GAUTAM V. PENDSE
%               DATE: 29th March 2010
%=========================================================================
%
%=========================================================================
%               PURPOSE:
%   Let f(w) = PG*(b-A*w) - (w'*b - (w'*A*w))*w
%   For a given vector w, the output will also be a q by 1 vector. We would
%   like this vector to be zero. In fact, we will use this function in the
%   code solve_for_w to perform a newton iteration to get w such that f(w)
%   = 0. This code will return the analytical Jacobian of f(w) at a given value of w.
%=========================================================================
%
%=========================================================================
%               INPUTS:
%    PG, b and A are quantities independent of w
%    w is the vector at which we wish to evaluate f(w)
%=========================================================================

    % define delta for numerical gradients
    delta = eps^(1/3);
    
    % get length of w
    q = length(w);
    
    
    % initialize Jacobian
    J = zeros(q,q);
    
    J = -PG*A - w*b' - b'*w*eye(length(w)) + w'*A*w * eye(length(w)) + 2*w*w'*A;
    
%     
%     for k = 1:q
%        % define w2
%        w2 = w;
%        w2(k) = w2(k) + delta;
%        
%        % define w1
%        w1 = w;
%        w1(k) = w1(k) - delta;
%        
%        J(:,k) = ( w_equation( PG, b, A, w2 ) - w_equation( PG, b, A, w1 ) )/2/delta;
%        
%     end

    %f = PG*(b-A*w) - (w'*b - (w'*A*w))*w;    

end


function found = check_convergence( w, pic, mu, sigma2, w_old, pic_old, mu_old, sigma2_old, tol )
%=========================================================================
%               AUTHOR: GAUTAM V. PENDSE
%               DATE: 26th March 2010
%=========================================================================
%
%=========================================================================
%               PURPOSE:
%   Check convergence
%=========================================================================
%
%=========================================================================
%               INPUTS:
%    w = random projection vector such that w'*w = 1 and G'*w = 0
%    pic = cell array of class fractions for the R classes
%    mu = cell array of class means for the R classes
%    sigma2 = cell array of variances for the R classes
%
%    w_old = old estimate: projection vector such that w'*w = 1 and G'*w = 0
%    pic_old = old estimate: cell array of class fractions for the R classes
%    mu_old = old_estimate: cell array of class means for the R classes
%    sigma2_old = old_estimate: cell array of variances for the R classes
%
%    tol = convergence tolerance
%=========================================================================

    % initialize flag to converged
    found = 1;

    % number of mixtures
    R = length(mu);

    % first make sure w_old and w have the first element positive
    if ( w(1) < 0 )
        w = -1*w;
    end
    
    if ( w_old(1) < 0 )
        w_old = -1*w_old;
    end
    
    % next make sure mu{1} and mu_old{1} are positive
    if ( mu{1} < 0 )
       for k = 1:R
          mu{k} = -1*mu{k}; 
       end
    end
    
    if ( mu_old{1} < 0 )
       for k = 1:R
          mu_old{k} = -1*mu_old{k}; 
       end
    end
    
    % Why do this? Because the likelihood depends on (w'*Z - mu_k)^2 = (mu_k - W'*Z)^2
    % Thus flipping the sign of w and mu{k} we get the same likelihood!
    
    % check w
    if ( max(abs(w - w_old)) > tol )
        found = 0;
    end    
    
    % check pic
    for k = 1:R
        
        if ( max(abs(pic{k} - pic_old{k})) > tol )
            found = 0;
        end
        
    end
    
    % check mu
    for k = 1:R
       
        if ( max(abs(mu{k} - mu_old{k})) > tol )
            found = 0;
        end
        
    end
    
    % check sigma2
    for k = 1:R
       
        if ( max(abs(sigma2{k} - sigma2_old{k})) > tol )
            found = 0;
        end
       
    end
    
end


function o = initialize_parameters_kmeans( Z, R, G )
%=========================================================================
%               AUTHOR: GAUTAM V. PENDSE
%               DATE: 26th March 2010
%=========================================================================
%
%=========================================================================
%               PURPOSE:
%   Initialize parameters for projected MOG
%=========================================================================
%
%=========================================================================
%               INPUTS:
%   Z = q by n matrix of vectors in R^q with q < n
%   R = number of components in the projected MOG
%   G = q by L matrix with L < q (to be used for imposing G^T w = 0
%   constraint) 
%=========================================================================
%
%=========================================================================
%               OUTPUTS:
%    o.w = random projection vector such that w'*w = 1 and G'*w = 0
%    o.pic = cell array of class fractions for the R classes
%    o.mu = cell array of class means for the R classes
%    o.sigma2 = cell array of variances for the R classes
%=========================================================================

    % get size of G
    q = size(Z, 1);
    
    % generate a random q by 1 vector w
    w = rand(q,1);
    
    % orthogonalize w.r.t G
    if ( isempty(G) == 1 )
        w = w;
    else
        w = w - G*inv(G'*G)*G'*w;
    end
    % unit normalize w
    w = w/norm(w);
    
    
    % compute projected points
    u = w'*Z; % 1 by n vector
    
    % run k-means with R clusters on u
    [IDX,C] = kmeans( u', R ); % C is an R by 1 matrix
    
    % initialize mixing proportions
    for k = 1:R
        pic{k} = sum(IDX == k)/length(IDX);
    end
    
    % initialize class means
    for k = 1:R
       mu{k} = C(k); 
    end
    
    % initialize class variances
    for k = 1:R
       sigma2{k} = var( u( find(IDX == k) ) );
    end
    
    o.w = w;
    o.pic = pic;
    o.mu = mu;
    o.sigma2 = sigma2;
           
end

function o = initialize_parameters_kmeans2( Z, R, G, beta, sigma2_a, sigma2_b, wuser )
%=========================================================================
%               AUTHOR: GAUTAM V. PENDSE
%               DATE: 26th March 2010
%=========================================================================
%
%=========================================================================
%               PURPOSE:
%   Initialize parameters for projected MOG
%=========================================================================
%
%=========================================================================
%               INPUTS:
%   Z = q by n matrix of vectors in R^q with q < n
%   R = number of components in the projected MOG
%   G = q by L matrix with L < q (to be used for imposing G^T w = 0 constraint) 
%   beta = R by 1 vector of prior parameters for the class fractions (user specified)
%   sigma2_a = R by 1 vector of the "a" parameter for the inverse gamma prior on sigma2{k}
%   sigma2_b = R by 1 vector of the "b" parameter for the inverse gamma prior on sigma2{k}
%=========================================================================
%
%=========================================================================
%               OUTPUTS:
%    o.w = random projection vector such that w'*w = 1 and G'*w = 0
%    o.pic = cell array of class fractions for the R classes
%    o.mu = cell array of class means for the R classes
%    o.sigma2 = cell array of variances for the R classes
%=========================================================================

    % do nmax iterations of w estimation and parameter estimation
    nmax = 5;
    
    % initialize count
    count = 1;
    
    % initialize stopping flag
    found = 0;
    
    %=========================================================================
    %                       initialize w
    % get size of G
    q = size(Z, 1);

    if ( isempty(wuser) == 0 )
        w = wuser;
    else
        % generate a random q by 1 vector w
        w = rand(q,1);
    end
    %=========================================================================    
    
    while( found == 0 )

        %=========================================================================
        %               FIRST get parameters for a given w
        

            % orthogonalize w.r.t G
            if ( isempty(G) == 1 )
                w = w;
            else
                w = w - G*inv(G'*G)*G'*w;
            end
            % unit normalize w
            w = w/norm(w);


            % compute projected points
            u = w'*Z; % 1 by n vector

            % run k-means with R clusters on u
            [IDX,C] = kmeans( u', R ); % C is an R by 1 matrix

            % initialize mixing proportions
            for k = 1:R
                pic{k} = sum(IDX == k)/length(IDX);
            end

            % initialize class means
            for k = 1:R
               mu{k} = C(k); 
            end

            % initialize class variances
            for k = 1:R
               sigma2{k} = var( u( find(IDX == k) ) );
            end    
        %=========================================================================
        
        %=========================================================================
        %               NEXT, get w using select_good_seed_point for given parameters
        
            w = select_good_seed_point(Z, G, pic, mu, sigma2, beta, sigma2_a, sigma2_b);
        %=========================================================================
        
        compute_likelihood( Z, w, pic, mu, sigma2, beta, sigma2_a, sigma2_b );
        pause(0.1);
        
        if ( count > nmax )
            found = 1;
        end
        
        % update counter
        count = count + 1;
        
    end
    
    
    o.w = w;
    o.pic = pic;
    o.mu = mu;
    o.sigma2 = sigma2;
           
end

