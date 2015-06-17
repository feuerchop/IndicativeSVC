% better iSvc by given both outliers and normals
clc;close all;
% generate data: 1 ring + 2 overlapped gaussian
addpath('../');
load syndata;
% settings
small_idx=randsample(length(Y),200);
data.X=X(small_idx,:);
data.y=Y(small_idx);
p1 = 0.2; p2=0.01;
c1 = min(1/(size(data.X,1)*p1), 1);
c2 = min(1/(size(data.X,1)*p2), 1);
h = 1;
ft = 18; % fontsize for title
%c = 1;
opt1=struct('method','CG','ker','rbf', 'arg',h, 'C', c1);
opt2=struct('method','CG','ker','rbf', 'arg',h, 'C', c2);

outlier = find(data.y==1);
cls_a = find(data.y==2);
cls_b = find(data.y==3);
normal = [cls_a;cls_b];

% options.Xr = normal_indic;
% options.Xa = outlier_indic;
% %% normal svc
% figure(1);
% ax=subplot(131);
% gf=spread(ax,X,Y,'',40);
% title(ax,'Original sample distribution','FontSize',ft);
% legend(ax,gf,{'Outliers','Class A','Class B'},'Location','NorthEast');
% % ax=subplot(132);
% % prob=gkde2(X);
% % colormap(ax,'Jet');
% % mesh(ax,prob.x,prob.y,prob.pdf);
% % title(ax,'Kernel Density estimation','FontSize',ft);
textpos_x = -13.5; textpos_y = 13; point_size = 6;
ax=subplot(121);
bx=subplot(122);
opt1.axe = ax;
opt2.axe = bx;
options.Xr = [];
options.Xa = [];
model1=semisvc(data,opt1); 
model2=semisvc(data,opt2);
plotNormalBound(ax,data,model1,point_size);
plotNormalBound(bx,data,model2,point_size);
%tle='iSVC cluster boundary';
%cap=['(c) C=',num2str(c,2),', h=',num2str(h,2)];
%ylabel(ax,cap,'FontSize',ft+4);
%text(textpos_x,textpos_y,tle,'FontSize',ft);
pause();
for rate=1:50
    cla(ax);cla(bx);
    idx=randsample(length(data.y),round(rate*length(data.y)/100));
    opt1.Xr=intersect(normal,idx);
    opt1.Xa=intersect(outlier,idx);
    opt2.Xr=opt1.Xr;
    opt2.Xa=opt1.Xa;
    model1=semisvc(data,opt1); 
    model2=semisvc(data,opt2);
    plotNormalBound(ax,data,model1,point_size);
    plotNormalBound(bx,data,model2,point_size);
    pause(0.2);
end