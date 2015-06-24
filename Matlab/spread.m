function handles=spread(ax, X, label, t, s)
% Plot samples of different labels with different colors.
% Written by Michael Chen (sth4nth@gmail.com).
[n, d] = size(X);
if nargin < 2
    label = ones(n,1);
end
assert(n == length(label));
% color = 'brkcmgy';
color = colormap(ax,'Jet');
% m = size(color,1);
handles=[];
% color = color(1:10:m, :);
% color(1,:) = color(end,:);
c = max(label);
if nargin < 4
    t = '';
end
if nargin < 5
    s=7;
end
% figure(gcf);
hold(ax, 'on');
axis(ax, 'equal');
axis(ax, 'square');
box(ax, 'on');
xlim(ax, [min(X(:,1))-1, max(X(:,1))+1]);
ylim(ax, [min(X(:,2))-1, max(X(:,2))+1]);
title(ax, t);
switch d
    case 2
        view(2);
        for i = 1:c
            %idc = label==i;
            if i == 1
                h=  plot(X(label==i, 1),X(label==i, 2),'o', 'Color',color(end,:),...
                    'MarkerSize',s,'MarkerFaceColor',color(end,:));
            else
                h = plot(X(label==i, 1),X(label==i, 2),'o', 'Color',color(5+5*i,:),...
                    'MarkerSize',s,'MarkerFaceColor',color(5+5*i,:));
                %h = scatter(ax, X(idc, 1),X(idc, 2),s,color(mod(i-1,m)+1,:),'.');
            end
            handles = [handles; h];
        end
    case 3
        view(3);
        for i = 1:c
            %idc = label==i;
            % plot3(X(idc, 1),X(idc, 2),X(idc, 3),['.' idc],'MarkerSize',15);
            h = scatter3(ax, X(idc, 1),X(idc, 2),X(idc, 3),s,color(mod(i-1,m)+1),'.');
            handles = [handles; h];
        end
    otherwise
        error('ERROR: only support data of 2D or 3D.');
end
