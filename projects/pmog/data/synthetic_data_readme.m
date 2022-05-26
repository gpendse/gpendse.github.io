%% Description of Synthetic Data 
function z = synthetic_data_readme()
%% Copyright Notice
%=========================================================================
%   COPYRIGHT NOTICE AND LICENSE INFO:
%
%   * Copyright (C) 2010, Gautam V. Pendse, McLean Hospital
%   
%   * Readme file for synthetic data contained in synthetic_pmog_data.mat used in the paper:
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



%% Explanation of the structure fields in the file synthetic_pmog_data.mat
%
% * |nruns| - number of runs of PMOG and FICA 
%
% * |nsources| - number of sources in the BSS mixture
%
% * |cmsm| - structure returned by create_mog_source_mixture.m
%
% * |S| - matrix of MOG souces. Each row is one source
%
% * |best_match_fica_for_source| - best match correlation coefficients between true sources and FICA estimated sources across runs
%
% * |best_match_pmog_for_source| - best match correlation coefficients between true sources and PMOG estimated sources across runs
%
% *The following variables change in each run* 
%%
% 
% * |A| - mixing matrix, a different mixing matrix is used for each run
%
% * |X| - mixed sources using the current mixing matrix
%
% * |fica| - sources returned by FICA in the current iteration
%
% * |mat_array_fica| - input for raicar_type_sorting_mat.m
%
% * |rcr_fica| - structure returned by raicar_type_sorting_mat.m
%
% * |pmi| - sources returned by PMOG in current iteration
%
% * |mat_array_pmog| - input for raicar_type_sorting_mat.m
%
% * |rcr_pmog| - structure returned by raicar_type_sorting_mat.m 



