function plotDist2Center( ax, data, model )
%PLOTDIST2CENTER Summary of this function goes here
%   plot distances of each point to its SVC center
    h = model.options.arg;
    x_ub = max(data.X(:,1))+8*h; x_lb = min(data.X(:,1))-8*h; xrange=x_ub-x_lb;
    y_ub = max(data.X(:,2))+8*h; y_lb = min(data.X(:,2))-8*h; yrange=y_ub-y_lb;
    xlim(ax,[x_lb x_ub]);
    ylim(ax,[y_lb y_ub]);
    axis(ax,'square');
    box(ax,'on');
    hold(ax,'on');
    grid = 50;
    [Ax,Ay] = meshgrid(linspace(x_lb,x_ub,grid),...
        linspace(y_lb,y_ub,grid));
    dist = kdist2([Ax(:)';Ay(:)'],model);
    dist = reshape(dist, grid, grid);
    colormap('hsv');
    mesh(ax,Ax,Ay,dist);
    title(ax,'Distances to SVC center');
end

