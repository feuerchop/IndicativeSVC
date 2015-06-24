function plotIndicatives( data, model )
%PLOT Summary of this function goes here
%   plot impact range of indicative points
    hold on;
    if ~isempty(model.iXr)
        % normal
        for i = 1:length(model.options.Xr)
            X = data.X(model.options.Xr(i),:);
            nodes = data.X(model.xrn(:,i)==1,:);
            A = [X; nodes];
            adj_mat = [ones(1, length(A)); zeros(length(A)-1, length(A))];
            gplot(adj_mat,A,'c-');
        end
    end
    if ~isempty(model.iXa)
        % outliers 
        for i = 1:length(model.options.Xa)
            X = data.X(model.options.Xa(i),:);
            nodes = data.X(model.xan(:,i)==1,:);
            A = [X; nodes];
            adj_mat = [ones(1, length(A)); zeros(length(A)-1, length(A))];
            gplot(adj_mat,A,'r-');
        end
    end
end

