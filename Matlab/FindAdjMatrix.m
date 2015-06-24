function [adjacent] = FindAdjMatrix(X,model)
%==========================================================================
% FindAdjMatrix: Caculating adjacency matrix
%
% Input:
%  X [dim x num_data] Input data.
%  model [struct] obtained from svdd.m 
%
% Output:
%  adjacent [num_data x num_data] 
%     1 for connected, 0 for disconnected (violated), -1 (outliers, BSV)
%
% Description
%	The Adjacency matrix between pairs of points whose images lie in
%	or on the sphere in feature space. 
%	(i.e. points that belongs to one of the clusters in the data space)
%
%	given a pair of data points that belong to different clusters,
%	any path that connects them must exit from the sphere in feature
%	space. Such a path contains a line segment of points y, such that:
%	kdist2(y,model)>model.r.
%	Checking the line segment is implemented by sampling a number of 
%   points (10 points).
%	
%	BSVs are unclassfied by this procedure, since their feature space 
%	images lie outside the enclosing sphere.( adjcent(bsv,others)=-1 )
%
%==========================================================================
% January 13, 2009
% Implemented by Daewon Lee
% WWW: http://sites.google.com/site/daewonlee/
%==========================================================================

% samples are column vectors

N=size(X,2);
adjacent = zeros(N);
R=model.r+10^(-7);  % Squared radius of the minimal enclosing ball

for i = 1:N %rows
   for j = 1:N %columns
       % if the j is adjacent to i - then all j adjacent's are also adjacent to i.
       if j<i
         if (adjacent(i,j) == 1)
              adjacent(i,:) = (adjacent(i,:) | adjacent(j,:));
          end
       else
            % if adajecancy already found - no point in checking again
            if (adjacent(i,j) ~= 1)
                % goes over 10 points in the interval between these 2 Sample points
                adj_flag = 1; % unless a point on the path exits the shpere - the points are adjacnet
                for interval = 0:0.1:1
                    z = X(:,i) + interval * (X(:,j) - X(:,i));
                    % calculates the sub-point distance from the sphere's center 
                    d = kdist2(z, model);
                    if d > R
                       adj_flag = 0;
                       break;
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
 























