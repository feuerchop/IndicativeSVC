function d = genRingData( c, r, w, n )
% generate a ring-like data set
% data are uniformly distributed along a ring with radiu r
    alpha = 360*rand(n,1);
    radius = ones(n,1)*r + (-w+2*w*rand(n,1));
    x = ones(n,1)*c(1)+radius.*sind(alpha);
    y = ones(n,1)*c(2)+radius.*cosd(alpha);
    d = [x y];
end

