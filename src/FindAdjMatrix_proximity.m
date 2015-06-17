%==========================================================================
%   FindAdjMatrix From Proximity Graph
%
%   Proximity graph-based labeling_method
%   -1: (DD) Delaunay Diagram Labeling Metho
%   -2: (MST) Minimum Spanning Tree Labeling Method
%   -3: (KNN) K-Nearest Neighbor Labeling Method
%
% Reference
%  Yang, V. Estivill-Castro, S.K. Chalup, Support Vector Clustering 
%  Through Proximity Graph Modelling, Proc. 9th Int¡¯l Conf. Neural 
%  Information Processing, 898-903, 2002.
%
%==========================================================================
% January 13, 2009
% Implemented by Daewon Lee
% WWW: http://sites.google.com/site/daewonlee/
%==========================================================================

function [adjacent] = FindAdjMatrix_proximity(X,result,model)

N=size(X,2);
adjacent = zeros(N);

R=model.r+10^(-7);

%% DD
if isequal(model.options.method,'DD') % DD
    for m=1:size(result,1)
        temp=[1 2;2 3;1 3];
        for k=1:3
           a=temp(k,:);
           i=result(m,a(1));j=result(m,a(2));
           if (adjacent(i,j) ~= 1)
               % goes over 10 points in the interval between these 2 Sample points
               adj_flag = 1; % unless a point on the path exits the shpere - the points are adjacnet
               for interval = 0:0.1:1
                   z = X(:,i) + interval * (X(:,j) - X(:,i));
                   % calculates the sub-point distance from the sphere's center
                   d = kdist2(z, model);
                   if d > R
                       adj_flag = 0;
                       interval = 1;
                   end
               end
               if adj_flag == 1
                   adjacent(i,j) = 1;
                   adjacent(j,i) = 1;
               end
           end
       end
   end
end

%% MST
if isequal(model.options.method,'MST') % MST
    for m=1:size(result,1)
       i=result(m,1);j=result(m,2);
       if (adjacent(i,j) ~= 1)
           % goes over 10 points in the interval between these 2 Sample points
           adj_flag = 1; % unless a point on the path exits the shpere - the points are adjacnet
           for interval = 0:0.1:1
               z = X(:,i) + interval * (X(:,j) - X(:,i));
               % calculates the sub-point distance from the sphere's center
               d = kdist2(z, model);
               if d > R
                   adj_flag = 0;
                   interval = 1;
               end
           end
           if adj_flag == 1
               adjacent(i,j) = 1;
               adjacent(j,i) = 1;
           end
       end
    end
end
%% KNN
if isequal(model.options.method,'KNN')    % KNN
    k=size(result,2);
    for m=1:size(result,1)
        for s=1:k
           i=m; j=result(m,s);
           if (adjacent(i,j) ~= 1)
               % goes over 10 points in the interval between these 2 Sample points
               adj_flag = 1; % unless a point on the path exits the shpere - the points are adjacnet
               for interval = 0:0.1:1
                   z = X(:,i) + interval * (X(:,j) - X(:,i));
                   % calculates the sub-point distance from the sphere's center
                   d = kdist2(z, model);
                   if d > R
                       adj_flag = 0;
                       interval = 1;
                   end
               end
               if adj_flag == 1
                   adjacent(i,j) = 1;
                   adjacent(j,i) = 1;
               end
           end
        end
    end
end

%% To connect nodes which can be connected via directly connected edges.
for i=1:N
    for j=1:i
        if (adjacent(i,j) == 1)
            adjacent(i,:) = (adjacent(i,:) | adjacent(j,:));
        end
    end
end
    
 
