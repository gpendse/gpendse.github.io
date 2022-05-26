%% Description of Image Data
function z = image_data_readme()
%% Copyright Notice
%=========================================================================
%   COPYRIGHT NOTICE AND LICENSE INFO:
%
%   * Copyright (C) 2010, Gautam V. Pendse, McLean Hospital
%   
%   * Readme file for image data contained in image_pmog_data1.mat and image_pmog_data2.mat used in the paper:
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



%% Explanation of the structure fields in the files image_pmog_data1.mat and image_pmog_data2.mat
%
% * |s| - sources created from BSD data http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/ after pre-processing by preprocess_image.m. Each row is one source
%
% * |mumix| - mean vector used to create mixed sources
%
% * |Amix| - mixing matrix used to create mixed sources

