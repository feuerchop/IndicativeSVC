function weight = ptsweight( X, c, Xr, Xa )
%PTSWEIGHT Summary of this function goes here
%   calculate the C vector for training points given postive and negative
%   samples indexs
    n = size(X,1);
    if isempty(Xr)
        xac = mean(X(Xa,:),1);
        dist2ac = sqrt(sum((X - repmat(xac, n, 1)).^2,2));
        [~, max_ind] = max(dist2ac);
        xrc = X(max_ind,:);
        dist2rc = sqrt(sum((X - repmat(xrc, n, 1)).^2,2));
        weight = c*dist2ac./(dist2ac+dist2rc) + (1/n)*(dist2rc./(dist2ac+dist2rc));
        return;
    end
    if isempty(Xa)
        xrc = mean(X(Xr,:),1);
        dist2rc = sqrt(sum((X - repmat(xrc, n, 1)).^2,2));
        [~, max_ind] = max(dist2rc);
        xac = X(max_ind,:);
        dist2ac = sqrt(sum((X - repmat(xac, n, 1)).^2,2));
        weight = c*dist2ac./(dist2ac+dist2rc) + (1/n)*(dist2rc./(dist2ac+dist2rc));
        return;
    end
    xrc = mean(X(Xr,:),1); xac = mean(X(Xa,:),1);
    dist2rc = sqrt(sum((X - repmat(xrc, n, 1)).^2,2));
    dist2ac = sqrt(sum((X - repmat(xac, n, 1)).^2,2));
    weight = c*dist2ac./(dist2ac+dist2rc) + (1/n)*(dist2rc./(dist2ac+dist2rc));
end

