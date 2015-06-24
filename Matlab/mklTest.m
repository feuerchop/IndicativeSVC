clear all; close all;
clc;
load ring.mat;
% --- data set 1: 2 rings + 1 gaussian outliers ---
n1 = 100; o1 = 10;
x1_1 = genRingData( [0,0], 3, 0.1, n1 ); y1_1 = ones(n1, 1);
x1_2 = genRingData( [0,0], 1.8, 0.1, n1 ); y1_2 = 2*ones(n1, 1);
x1_ot = mvnrnd([0, -10], [0.1 0.1], 10); y1_ot = 3*ones(o1, 1);
X1 = [x1_1; x1_2; x1_ot]; Y1 = [y1_1; y1_2; y1_ot];
% default 
options=struct('method','CG','ker','mker','kerw', [0.5,0.5], 'arg',0.8,'C',0.1);
data.X=X1;   % data.X: p x N input patterns
ids = find(Y1==3);
outlier_idx = ids(1:5);
%% multiple kernel
% no labels 
options.ker = 'mker'; options.arg=0.8;
ax = subplot(2,3,4);
options.axe = ax;
[model,gf]=semisvc(data,[],[],options); hold on;
title(ax,'rbf+linear | no labels | sig=0.8 | coef=[0.5 0.5]');
%plot(options.axe,data.X(outlier_idx,1),data.X(outlier_idx,2), 'rs', 'MarkerSize', 10);
% with positive labels 
ax = subplot(2,3,5);
options.axe = ax;
[model,gf]=semisvc(data,[],outlier_idx,options); hold on;
plot(options.axe,data.X(outlier_idx,1),data.X(outlier_idx,2), 'rs', 'MarkerSize', 10);
title(ax,'rbf+linear | with labels | sig=0.8 | coef=[0.5 0.5]');

% tune sigma
ax = subplot(2,3,6);
options.axe = ax; options.arg = 0.8; options.kerw=[0.8 0.2];
[model,gf]=semisvc(data,[],outlier_idx,options); hold on;
plot(options.axe,data.X(outlier_idx,1),data.X(outlier_idx,2), 'rs', 'MarkerSize', 10);
title(ax,'rbf+linear | with labels | sig=0.8 | coef=[0.8 0.5]');
