function [ options, iXr, iXa, xr_neigh, xa_neigh ] = influc2( K, options)
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
    % find impacted points separately
    if isempty(options.Xa)
        norm_Kr = K(:, options.Xr);
        norm_Kr(norm_Kr < exp(-2)) = 0;
        infect_num_xr = sum(ceil(norm_Kr),2);
        influc_xr = -sum(norm_Kr,2)./infect_num_xr;
        influc_xr(isnan(influc_xr))=0;
        influc_vec = influc_xr;
    else
        if isempty(options.Xr)
            norm_Ka = K(:, options.Xa);
            norm_Ka(norm_Ka < exp(-2)) = 0;
            infect_num_xa = sum(ceil(norm_Ka),2);
            influc_xa = sum(norm_Ka,2)./infect_num_xa;
            influc_xa(isnan(influc_xa))=0;
            influc_vec = influc_xa;
        else
            norm_Kr = K(:, options.Xr);
            norm_Kr(norm_Kr < exp(-2)) = 0;
            infect_num_xr = sum(ceil(norm_Kr),2);
            influc_xr = -sum(norm_Kr,2)./infect_num_xr;
            influc_xr(isnan(influc_xr))=0;
            norm_Ka = K(:, options.Xa);
            norm_Ka(norm_Ka < exp(-2)) = 0;
            infect_num_xa = sum(ceil(norm_Ka),2);
            influc_xa = sum(norm_Ka,2)./infect_num_xa;
            influc_xa(isnan(influc_xa))=0;
            influc_vec = influc_xa+influc_xr;
        end
    end
    influc_vec(find(abs(influc_vec) < exp(-2))) = 0;
    iXr = find(influc_vec < 0);
    iXa = find(influc_vec > 0);
    % normal points are reweighted at the range [C~1]
    options.C(iXr)= options.C(iXr).*(1 + influc_vec(iXr))/(1-exp(-2)) + ...
        (-influc_vec(iXr) - exp(-2))/(1-exp(-2));
    % normal points are reweighted at the range [1/N~C]
    options.C(iXa) = options.C(iXa).*(1 - influc_vec(iXa))/(1-exp(-2)) + ...
        (influc_vec(iXa) - exp(-2))/(N*(1-exp(-2)));
end

end

