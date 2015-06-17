function h = estBandwidth( X )
%ESTBANDWIDTH Summary of this function goes here
%   estimate rbf kernel bandwidth
    n = size(X,1);
    n0 = min(200, n);
    x0 = X(randsample(n,n0),:);
%     h = 2*sum(pdist(x0,'seuclidean'))/(n0*(n0-1));
    h = norm(std(x0,1,1));
end

