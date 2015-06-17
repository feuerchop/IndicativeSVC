clc;close all;clear;
addpath(genpath('../../od2/'));
d=load('winequality-red.mat');
x=d.data(:,1:11);
y=d.data(:,12);
% 3,4,7,8 are outliers -> 280 samples
y(find(y<=4 | y>=7))=1;
% others are normal -> 1319 samples
y(find(y~=1))=0;
data.X=x;
data.y=y;
save('../wine-red-all.mat','data');