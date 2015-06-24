function d=kdist2(X,model)

%==========================================================================
% KDIST2 Computes squared distance between vectors in kernel space.
%
% Synopsis:
%  d = kdist2(X,model)
%
% Description:
%  It computes distance between vectors mapped into the feature
%  space induced by the kernel function (model.options.ker,
%  model.options.arg). The distance is computed between images
%  of vectors X [dim x num_data] mapped into feature space
%  and a point in the feature space given by model:
%
%   d(i) = kernel(X(:,i),X(:,i))
%          - 2*kernel(X(:,i),models.sv.X)*model.Alpha + b,
%
%  where b [1x1] is assumed to be equal to
%   model.b = model.Alpha'*kernel(model.sv.X)*model.Alpha.
%
% Input:
%  X [dim x num_data] Input vectors.
%  model [struct] Deternines a point of the feature space:
%   .Alpha [nsv x 1] Multipliers.
%   .sv.X [dim x nsv] Vectors.
%   .b [1x1] Bias.
%   .options.ker [string] Kernel identifier (see 'help kernel').
%   .options.arg [1 x nargs] Kernel argument(s).
%
% Output:
%  d [num_data x 1] Squared distance between vectors in the feature space.
%
%==========================================================================
% January 13, 2009
% Implemented by Daewon Lee
% WWW: http://sites.google.com/site/daewonlee/
%==========================================================================

[dim,num_data]=size(X);
if isequal(model.options.ker, 'mker')
    x2 = diag(mker(model.options.kerw,model.options.arg,X'));
    Ksvx = mker(model.options.kerw, model.options.arg,X',model.sv.X);
    d = x2 - 2*Ksvx*model.Alpha(:) + model.b*ones(num_data,1) ;
else
    x2 = diagker( X, model.options.ker, model.options.arg);
    Ksvx = kernel( X, model.sv.X', model.options.ker, model.options.arg);
    d = x2 - 2*Ksvx*model.Alpha(:) + model.b*ones(num_data,1) ;
end
