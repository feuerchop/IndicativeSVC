function varargout=plotsvc(ax,data,model,Xr,Xa)

x_ub = max(data.X(:,1)); x_lb = min(data.X(:,1)); xrange=x_ub-x_lb;
y_ub = max(data.X(:,2)); y_lb = min(data.X(:,2)); yrange=y_ub-y_lb;
xlim(ax,[x_lb-0.1*xrange x_ub+0.1*xrange]);
ylim(ax,[y_lb-0.1*yrange y_ub+0.1*yrange]);
axis(ax,'square');
box(ax,'on');
grid = 100;
[Ax,Ay] = meshgrid(linspace(x_lb-0.1*xrange,x_ub+0.1*xrange,grid),...
    linspace(y_lb-0.1*yrange,y_ub+0.1*yrange,grid));
dist = kdist2([Ax(:)';Ay(:)'],model);
gca
if isequal(model.options.method,'SEP-CG') | isequal(model.options.method,'E-SVC')
    %figure;
    hold(ax, 'on');
    test.X=data.X';
    test.y=model.cluster_labels;
    h1=ppatterns2(data.X(model.sv_ind,:)','ro',10);   % SV
    h2=plot(ax, model.local(1,:),model.local(2,:),'rs','MarkerSize',10); % SEP
    if isfield(model,{'ts'})
        h3=plot(ax, model.ts.x(:,1),model.ts.x(:,2),'r+','MarkerSize',10);
        legend(ax,[h1 h2 h3], 'SVs','SEPs','TSs','Location','SouthEast');
    else
        legend(ax,[h1,h2], 'SVs','SEPs','Location','SouthEast');
    end
    ppatterns2(test);
    contour(ax, Ax, Ay, reshape(dist,grid,grid),[model.r model.r],'k');
    title(ax, strcat(['Labelling method: ', model.options.method]));
else
    %figure;
    hold(ax, 'on');
    test.X=data.X(model.inside_ind,:)';
    test.y=model.cluster_labels(model.inside_ind);
    h1=ppatterns2(data.X(model.sv_ind,:)','ro',10);   % SV
    h2=ppatterns2(data.X(model.bsv_ind,:)','k.',12);  % BSV
    h3=ppatterns2(test);
    if ~isempty(h2)
        legstr={'SVs', 'BSV (outliers)'};
    else
        legstr={'SVs'};
    end
    for nh = 1:length(h3)
        legstr(length(legstr)+1) = {['class ' num2str(nh)]};
    end
    contour(ax, Ax, Ay, reshape(dist,grid,grid),[model.r model.r],'k');
    legend(ax, [h1,h2,h3],legstr,'Location','SouthEast');
    title(ax, strcat(['Labelling method: ', model.options.method]));
end
if nargin > 3
    if ~isempty(Xr)
        legstr(length(legstr)+1) = {'True normal'};
    end
    if ~isempty(Xa)
        legstr(length(legstr)+1) = {'True outlier'};
    end
    plot(ax,data.X(Xa,1),data.X(Xa,2), 'rs', 'MarkerSize', 10);
    plot(ax,data.X(Xr,1),data.X(Xr,2), 'gs', 'MarkerSize', 10);
end
hold(ax, 'off');


