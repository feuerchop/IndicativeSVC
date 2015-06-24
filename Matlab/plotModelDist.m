function plotModelDist( ax, data, model, options )
%PLOTMODELDIST Summary of this function goes here
%   plot model points distances to its center
    if nargin < 4
        options = 'r-';
    end
    dist = kdist2(data.X,model);
    plot(ax, 1:size(data.X,1), dist, options);
end

