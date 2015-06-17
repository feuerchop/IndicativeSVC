function [ options, iXr, iXa, xr_neigh, xa_neigh ] = influc( K, options)
%INFLUC Summary of this function goes here
% Influcence function assigns various penalty coefficient C accordingly
% Only rbf kernel is assumed
% INPUT:
% K - kernel matrix
% options - model options for learning SVC
% t: parameter for decay factor
% OUTPUT:
% options - options with reweighted penalty
% iXr - influenced regular data samples indexes
% iXa - influenced anomaly data samples indexes
N = size(K,1);
iXr = [];
iXa = [];
xr_neigh = [];
xa_neigh = [];
if ~isempty(options.Xa) || ~isempty(options.Xr)
    options.C = options.C*ones(N,1);
end

if ~isempty(options.Xr)
    % indicative normal set is not empty
    % find all influenced samples within 2*bandwidth
    norm_K = K(:, options.Xr);
    [r,c,~] = find(norm_K >= exp(-2));
    sparse_neigh = sparse(r,c,ones(length(r),1),size(norm_K,1),size(norm_K,2));
    xr_neigh = sparse_neigh;
    influc_neigh = sum(sparse_neigh, 2);
    influc_weight_Xr = sum(norm_K.*sparse_neigh,2);
    influc_weight_Xr(options.Xr) = 1; % reset normal indicators
    influc_neigh(options.Xr) = 1;
    iXr = find(influc_weight_Xr);
    options.C(iXr)= options.C(iXr).*(1 - influc_weight_Xr(iXr)./influc_neigh(iXr))/(1-exp(-2)) + ...
                    (influc_weight_Xr(iXr)./influc_neigh(iXr) - exp(-2))/(1-exp(-2));
end
if ~isempty(options.Xa)
    % indicative outlier set is not empty
    % find all influenced samples within 2*bandwidth
    anorm_K = K(:, options.Xa);
    [r,c,~] = find(anorm_K >= exp(-2));
    sparse_neigh = sparse(r,c,ones(length(r),1),size(anorm_K,1),size(anorm_K,2));
    xa_neigh = sparse_neigh;
    influc_neigh = sum(sparse_neigh, 2);
    influc_weight_Xa = sum(anorm_K.*sparse_neigh,2);
    influc_weight_Xa(options.Xa) = 1; % reset outlier indicators
    influc_neigh(options.Xa) = 1;
    iXa = find(influc_weight_Xa);
    options.C(iXa) = options.C(iXa).*(1 - influc_weight_Xa(iXa)./influc_neigh(iXa))/(1-exp(-2)) + ...
                    (influc_weight_Xa(iXa)./influc_neigh(iXa) - exp(-2))/(N*(1-exp(-2)));
    %options.C(iXa) = 1/N; 
end

end

