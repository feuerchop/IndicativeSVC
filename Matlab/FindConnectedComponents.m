function [clusters_assignments] = FindConnectedComponents(adjacent_matrix)

%==========================================================================
%
%	FindConnectedComponents
%	-----------------------
%
%	Parameters: 
%		adjacent_matrix 		- The adjacent matrix
%       N        		 		- The Number of samples.
%
%	Return Value:
%		clusters_assignments - label each point with its 
%							   cluster assignement.
%
%	Finds Connected Components in the graph represented by  
%	the adjacency matrix, each component represents a cluster.
%
%
%==========================================================================
% January 13, 2009
% Implemented by Daewon Lee
% WWW: http://sites.google.com/site/daewonlee/
%==========================================================================

% the clustering vector
N=size(adjacent_matrix,1);
clusters_assignments = zeros(N,1);
cluster_index = 0;

done = 0; % boolean - are we done clustering

while (done ~= 1)
   
   % select an un-clustered root for the DFS
   root = 1;
   while (clusters_assignments(root) ~= 0)
      root = root + 1;
      if (root > N) % all nodes are clustered
         done = 1;
         break;
      end
   end   
      
   % DFS
   % ===
   if (done ~= 1) % an unclustered node was found - start DFS
   
	   % a new cluster was found 
      cluster_index = cluster_index + 1;
      
      % the DFS stack
      stack = zeros(N,1);
      stack_index = 0;
      
      % put the root in the stack
      stack_index = stack_index + 1;
	  stack(stack_index) = root;   
   
	  % as long as the stack in not empty - continue
      while (stack_index ~= 0);
         
         % take a node from the stack
         node = stack(stack_index);
         stack_index = stack_index - 1;
         
         % assign the cluster number to it
         clusters_assignments(node) = cluster_index;
   
   		 % check all its neighbors
   		 for i = [1:N]
      
            % check that this node is a neighbor and not clustered yet
            if (adjacent_matrix(node,i) == 1 & ...
					clusters_assignments(i) == 0 & ...
                    i ~= node)
               % add to stack
               stack_index = stack_index + 1;
               stack(stack_index) = i;   
            end
         end
      end
   end
end   

