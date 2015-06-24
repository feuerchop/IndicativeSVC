function h=showDigits( X, col )
% test displaying digits
    X = X';
    m = size(X,2);
    img_mat = reshape(X(:), 28, 28, m);
    row = ceil(m/col);
    for n = 1:m
        ax=subplot(row,col,n);
        subimage(img_mat(:,:,n)');
        set(gca,'xtick',[],'ytick',[]);
    end
    h=gcf;
end

