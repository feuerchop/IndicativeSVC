function model = semisvdd( X, options )
% solve the reweighted Wolfe dual problem using cvx
% input:
% X: input data
% options: control parameters
% output: 
% svc model, see semisvc
% Problem to be solved:
% min W=x'*diag(K)-x'*K*x
% s.t. 0 <= x_i <= C_i
%      x_r=0
%      x_a = C_a

N = size(X,1);
if ~isfield(options,{'ker'})
    options.ker='rbf';
end
if isequal(options.ker, 'mker')
    % use multiple kernel here. See: mker
    K = mker(options.kerw, options.arg, X);
else
    K = kernel(X', options.ker, options.arg);
end
eps = 10^(-3);
% recompute C vector based on the indicative sample set
[options, iXr, iXa, xr_neigh, xa_neigh] = influc2(K, options);

% solve the problem use cvx
% cvx_begin
%     variable x(N)
%     minimize(-x'*diag(K)+x'*K*x)
%     %minimize(-x'*diag(K)+x'*K*x+sum(sum(X(Xa,:)*X'*x)));
%     subject to
%         0 <= x <= options.C
%         if ~isempty(options.Xr)
%             x(options.Xr) < options.C(options.Xr)-eps;
%         end
%         if ~isempty(options.Xa)
%             x(options.Xa) >= options.C(options.Xa)-eps;
%         end
% cvx_end
% matlab default qp solver
f = -diag(K);
H=2*K;
opt = optimset('Algorithm','interior-point-convex','Display','off');
if ~isempty(options.Xa) || ~isempty(options.Xr)
    % equality constrains only for outliers
%     neq = length([options.Xa; options.Xr]);
%     Aeq = sparse(1:neq,[options.Xa;options.Xr],ones(neq,1),neq,N);
%     beq = [options.C(options.Xa); zeros(length(options.Xr),1)];
    neq = length(options.Xa);
    Aeq = sparse(1:neq,options.Xa,ones(neq,1),neq,N);
    beq = options.C(options.Xa);
    [x,fval]=quadprog(H,f,[],[],Aeq,beq,zeros(N,1),options.C,[],opt);
else
    [x,fval]=quadprog(H,f,[],[],[],[],zeros(N,1),ones(N,1)*options.C,[],opt);
end
% setup model
inx= find(x > eps);
model.Alpha = x(inx);   % non-zero coefficients, zero indicates point-inside
model.sv_ind=find(x > eps & x < options.C-eps);   % indexes of support vectors
model.bsv_ind=find(x >= options.C-eps);   % indexes of bounded support vectors
model.inside_ind=find(x < options.C-eps);   % indexes of points inside the sphere
model.b = x(inx)'*K(inx,inx)*x(inx);
%--------------------
model.sv.inx = inx;
model.sv.X = X(model.sv.inx, :);
model.nsv = length(inx);
model.options = options;
model.r = max(kdist2(X(model.sv_ind,:)',model));
model.iXr = iXr;  % influcenced potential regular samples indexes
model.iXa = iXa;  % influcenced potential outliers indexes
model.xrn = xr_neigh; % sparse matrix indicating each given regular indicator's impacts
model.xan = xa_neigh; % sparse matrix indicating each given abnormal indicator's impacts
end
