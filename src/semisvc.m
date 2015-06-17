function [model, gh] = semisvc( data, options )
% Semi-supervised support vector clustering algorithm
% Description:
% In outlier detection scenarios, mostly users are given feedback when a
% detecting process is about to be conducted. These feedbacks converts a total
% unlabelled data as a partially labelled data? such that a semi-supervised
% learning scheme is very valuable in such kind of scenes. Traditional SVC
% assumes there's no prior knowledge about the data, which sometimes makes the
% outliers be misjudged or neglected. the semisvc algorithm make full use
% of the partly available user labels to improve the clustering accuracy,
% especially on the robust outlier detection.
% INPUT:
% [data]
%     .X: Nxd Training data set without labels
%     .Y: Nx1 Traning data labels
% [XR] rx1 vector for the indexes of regular data samples
% [XO] ax1 vector for the indexes of anomalies samples
% options *struct* for semiSVC
%   .ker [string] Kernel identifier (default 'linear'). See 'help kernel'.
%   .arg [1 x nargs] Kernel arguments.
%   .C scalar or [Nx1] Regularization constant (default []);
%    If C = 0 (or C=1, default), it is hard margin, that is,
%       the model do not allow bounded support vectors
%       it can be a Nx1 weight vector where each point is penalized differently      
%   .method [string] SVC cluster labeling Method
%       'CG': Complete Graph Labeling Method. See Ref. 1 (default)
%       'DD': Delaunay Diagram Labeling Method. See Ref. 4
%       'MST': Minimum Spanning Tree Labeling Method. See Ref. 4
%       'KNN': K-Nearest Neighbor Labeling Method. See Ref. 4
%               >> options.k (number of K, default=4)
%       'SEP-CG': SEP-based CG Labeling Method. See Ref. 2
%       'E-SVC': Equilibrium vector based Labeling Method. See Ref. 3
%          >> options.NofK (number of clusters, default = 0 (automatic determination) )
%          --> We can control the number of clusters of SVC directly!
%   .Xr [Rx1] Indexes of regular samples
%   .Xa [Ax1] Indexes of anomaly samples
% Output:
%  model [struct] Center of the ball in the kernel feature space:
%   .sv.X [dim x nsv] Data determining the center. (SV + BSV)
%   .Alpha [nsv x 1] Data weights. (SV + BSV)
%   .r [1x1] squared radius of the minimal enclosing ball. i.e. R^2
%   .b [1x1] Squared norm of the center equal to Alpha'*K*Alpha.
%   .sv_ind [nsv2 x 1] index of corresponding SV among X
%   .bsv_ind [nbsv x 1] index of corresponding BSV among X
%   .inside_ind [num_data-nbsv x 1] index of point inside the sphere
%   .options [struct] Copy of used options.
%   .stat [struct] Statistics about optimization:
%     .access [1x1] Number of requested columns of matrix H.
%     .t [1x1] Number of iterations.
%     .UB [1x1] Upper bound on the optimal value of criterion.
%     .LB [1x1] Lower bound on the optimal value of criterion.
%     .LB_History [1x(t+1)] LB with respect to iteration.
%     .UB_History [1x(t+1)] UB with respect to iteration.
%     .NA [1x1] Number of non-zero entries in solution.
%
% Author: Huang Xiao
% Institute: Technische Universitaet Muenchen
% Copyright: Chair of IT Security (I20)
% Date: 2013.05.01
% ------------------------------------
% Initialization
N = size(data.X,1); d = size(data.X,2);
if nargin < 2
    % set default options, default portion of outliers in data set is 0.05
    options=struct('method','CG','ker','rbf','arg',1,'C',1/(N*0.05));
end
if ~isfield(data,{'y'})
    data.y=ones(1,N);
end

if ~isfield(options,{'ker'})
    options.ker='rbf';
end

if ~isfield(options,{'kerw'})
    options.kerw=[0.5 0.5];
end

if ~isfield(options,{'arg'})
    options.arg=1;
end

if ~isfield(options,{'C'})
    options.C=1/(N*0.05);    % default portion of outliers in data set is 0.01
end

if ~isfield(options,{'method'})
    options.method='CG';
end

if isequal(options.method,'KNN') & ~isfield(options,{'k'})
    options.k=4;
end

if ~isfield(options,{'axe'})
    %options.axe = gca;
end

if ~isfield(options,{'Xr'})
    options.Xr = [];
end

if ~isfield(options,{'Xa'})
    options.Xa = [];
end
%% Optimzing SVDD model
disp('Step 1: Optimizing the restricted Wolfe dual problem.....');
model = semisvdd(data.X, options);
%% Cluster labelling
if nargout > 1
    disp(strcat(['Step 2: Labeling cluster index by using ',options.method,'....']));
    switch options.method
        case 'CG'
            adjacent = FindAdjMatrix(data.X(model.inside_ind,:)',model);
            % Finds the cluster assignment of each data point
            clusters = FindConnectedComponents(adjacent);
            model.cluster_labels=zeros(1,length(data.y));
            model.cluster_labels(model.inside_ind)=double(clusters);
        case 'DD'
            [result]=plot_dd(data.X(:,model.inside_ind)');
            adjacent = FindAdjMatrix_proximity(data.X(:,model.inside_ind),result,model);
            % Finds the cluster assignment of each data point
            clusters = FindConnectedComponents(adjacent);
            model.cluster_labels=zeros(1,length(data.y));
            model.cluster_labels(model.inside_ind)=double(clusters);
        case 'MST'
            [result]=plot_mst(data.X(:,model.inside_ind)');
            adjacent = FindAdjMatrix_proximity(data.X(:,model.inside_ind),result,model);
            % Finds the cluster assignment of each data point
            clusters = FindConnectedComponents(adjacent);
            model.cluster_labels=zeros(1,length(data.y));
            model.cluster_labels(model.inside_ind)=double(clusters);
        case 'KNN'
            [result]=knn_svc(options.k,data.X(:,model.inside_ind));
            adjacent = FindAdjMatrix_proximity(data.X(:,model.inside_ind),result,model);
            % Finds the cluster assignment of each data point
            clusters =  FindConnectedComponents(adjacent);
            model.cluster_labels=zeros(1,length(data.y));
            model.cluster_labels(model.inside_ind)=double(clusters);
            
        case 'SEP-CG'
            % find stable equilibrium points
            [rep_locals,locals,local_val,match_local]=FindLocal(data.X',model);
            model.local=locals';    % dim x N_local
            %small_n=size(locals,1);
            % calculates the adjacent matrix
            [adjacent] = FindAdjMatrix(model.local,model);
            % Finds the cluster assignment of each data point
            local_clusters_assignments = FindConnectedComponents(adjacent);
            model.cluster_labels = local_clusters_assignments(match_local)';
        case 'E-SVC'
            if ~isfield(model.options,{'epsilon'})
                model.options.epsilon=0.05;
            end
            if ~isfield(model.options,{'NofK'})
                model.options.NofK=0;
            end
            [model] = esvcLabel(data,model);
    end
    plotsvc(options.axe,data,model); 
    gh=options.axe;
else
    display('Cluster labelling skipped...');
end
disp('Finished SVC clustering!');
end

