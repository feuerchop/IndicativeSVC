function plotDigits( X, data, xid)
%PLOTDIGITS Summary of this function goes here
%   X is all data set, X0 is the images to be shown
    pos = data.X; label = data.y;
    xmean = mean(pos);
    eps = 0.8;
    dist = sqrt(sum((pos-repmat(xmean,length(pos),1)).^2,2));
    r = 0.8*max(dist);
    hold on;
    for i=1:length(xid)
        id = xid(i);
        img_mat = X(id,:); % 1x784 ubit8
        tahn = (pos(id,1)-xmean(1))/(pos(id,2)-xmean(2));
        img_pos_x = tahn*sqrt(r^2/(tahn^2+1))+xmean(1);
        img_pos_y = sqrt(r^2/(tahn^2+1))+xmean(2);
        moveit2(subimage([img_pos_x;img_pos_x+eps], [img_pos_y;img_pos_y-eps], reshape(img_mat,28,28)'));
    end
end

