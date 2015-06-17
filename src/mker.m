function k = mker( coef, sigma, X1, X2 )
% Self-defined multiple weighted kernel function: rbf+linear
% X1: Nxd data input
% X2: (optional) Mxd data input
% coef: 2x1 weight vector
% sigma: for rbf kernel

if nargin < 4
    n1sq = sum(X1.^2, 2);
    K_rbf = exp((repmat(n1sq, 1, size(X1, 1)) + (repmat(n1sq, 1, size(X1, 1)))' - 2*(X1*X1'))/(-2*sigma^2));
    K_linear = X1*X1';
    D1 = diag(1./sqrt(diag(K_rbf)));
    D2 = diag(1./sqrt(diag(K_linear)));
    k = coef(1)*D1*K_rbf*D1+coef(2)*D2*K_linear*D2;
else
%     n1sq = sum(X1.^2, 2);
%     n2sq = sum(X2.^2, 2);
%     K_rbf = exp((repmat(n1sq, 1, size(X2, 1)) + repmat(n2sq, 1, size(n1sq))' - 2*X1*X2')/(-2*sigma^2));
%     K_linear = X1*X2';
%     K_rbf = kernel(X1',X2','rbf',sigma);
%     K_linear = kernel(X1',X2','linear',1);
%     K1_rbf = kernel(X1','rbf',sigma);
%     K1_linear = kernel(X1', 'linear', 1);
%     K2_rbf = kernel(X2','rbf',sigma);
%     K2_linear = kernel(X2','linear',1);
%     D1 = diag(sqrt(coef*[diag(K1_rbf) diag(K1_linear)]'));
%     D2 = diag(sqrt(coef*[diag(K2_rbf) diag(K2_linear)]'));
%     k = D1*(coef(1)*K_rbf+coef(2)*K_linear)*D2;
    m = size(X1,1);
    n = size(X2,1);
    newX = [X1;X2];
    n1sq = sum(newX.^2,2);
    K_rbf = exp((repmat(n1sq, 1, size(newX, 1)) + (repmat(n1sq, 1, size(newX, 1)))' - 2*(newX*newX'))/(-2*sigma^2));
    K_linear =newX*newX';
    D1 = diag(1./sqrt(diag(K_rbf)));
    D2 = diag(1./sqrt(diag(K_linear)));
    kfl = coef(1)*D1*K_rbf*D1+coef(2)*D2*K_linear*D2;
    k = kfl(1:m,m+1:end);
end


