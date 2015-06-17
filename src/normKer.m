function k = normKer( K )
%NORMKER Summary of this function goes here
%   normalize a kernel matrix
  diagonal = diag(1./sqrt(diag(K)));
  k = diagonal*K*diagonal;
end

