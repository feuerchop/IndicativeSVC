function  plotLasers( X, neighbours )
%PLOTRADIAC Summary of this function goes here
%  Plot influenced points 
    A = [X; neighbours];
    adj_mat = [ones(1, length(A)); zeros(length(A)-1, length(A))];
    gplot(adj_mat,A,'m-');
end

