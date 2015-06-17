function mu = phLoss( X, sig )
% Pseudo-Huber Loss function
    mu = sig/2 + exp(ones(X,1)*sig - X)*sig/2;
end

