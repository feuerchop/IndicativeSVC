function [X, h] = addMnistNoise(X,w)
% add noise to MNIST digits
% X is a nx768 matrix, where each row is an image
% randomly pick up w cols in image and add randomly white noise on them
    for i=1:size(X,1)
        cols = randsample(28,w);
        A = repmat(28*cols',28,1);
        B = repmat([27:-1:0]',w,1);
        bad_pix_idx = A(:) - B(:);
        X(i,bad_pix_idx) = floor(255*rand(1,length(bad_pix_idx)));
    end
    if nargout > 1
        h=showDigits(X, 10);
    end
end
