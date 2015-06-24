function plotNormalBound( ax, data, model, pw )
%PLOTNORMALBOUND Summary of this function goes here
%   Detailed explanation goes here
% kernel bandwidth
if nargin < 4
    pw = 7;
end
Xa = model.options.Xa;
Xr = model.options.Xr;
h = model.options.arg;
x_ub = max(data.X(:,1))+h; x_lb = min(data.X(:,1))-h; xrange=x_ub-x_lb;
y_ub = max(data.X(:,2))+h; y_lb = min(data.X(:,2))-h; yrange=y_ub-y_lb;
xlim(ax,[x_lb x_ub]);
ylim(ax,[y_lb y_ub]);
color = colormap(ax,'Jet');
% m = size(color,1);
% color = color(1:10:m, :);
% color(1,:) = color(end,:);
axis(ax,'square');
box(ax,'on');
hold(ax,'on');
grid = 100;
[Ax,Ay] = meshgrid(linspace(x_lb,x_ub,grid),...
    linspace(y_lb,y_ub,grid));
dist = kdist2([Ax(:)';Ay(:)'],model);
dist = reshape(dist, grid, grid);
leg_str={};
% plot normal points
h_norm=plot(ax,data.X(model.inside_ind,1), data.X(model.inside_ind,2), 'o', 'Color', color(15,:),...
    'MarkerSize',pw,'MarkerFaceColor',color(15,:));
if ~isempty(h_norm) leg_str(length(leg_str)+1) = {'Normal digits \{2,6,9\}'}; end    
% plot support vectors
h_sv=plot(ax,data.X(model.sv_ind,1), data.X(model.sv_ind,2), 'o', 'Color',color(15,:), 'MarkerSize', 3*pw);
if ~isempty(h_sv) leg_str(length(leg_str)+1) = {'SVs'}; end 
% plot outliers
h_bsv=plot(ax,data.X(model.bsv_ind,1), data.X(model.bsv_ind,2), 'o', 'Color',color(end,:),...
    'MarkerSize',pw, 'MarkerFaceColor', color(end,:));
if ~isempty(h_bsv) leg_str(length(leg_str)+1) = {'Anomalies : Digit \{0\}'}; end 
% plot clustering boundry
contour(ax,Ax,Ay,dist,[model.r,model.r],'-k');
% plot user given pos/neg points
h_neg=[];
h_pos=[];
if ~isempty(Xr)
    h_neg=plot(data.X(Xr,1),data.X(Xr,2),'gs','MarkerSize',3*pw);
    leg_str(length(leg_str)+1) = {'Indicative digits \{2,6,9\}'}; 
end
if ~isempty(Xa)
    h_pos=plot(data.X(Xa,1),data.X(Xa,2),'rs','MarkerSize',3*pw);
    leg_str(length(leg_str)+1) = {'Indicative digit \{0\}'}; 
end
legend([h_norm,h_sv,h_bsv,h_neg,h_pos],leg_str,'Location','NorthEast','FontSize',16);
%title(gca,'SVC Boundry of Outliers/Normals');
end

