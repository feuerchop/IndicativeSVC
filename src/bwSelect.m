function [h,g] = bwSelect( data, opt )
% bandwidth selection for svc
if ~isfield(opt,'A')
    % TODO: now only svc is supported.
    error('No algorithm is selected! exit.');
    return;
end
if ~isfield(opt,'k')
    % default rep. value is 10
    opt.k = 10;
end
if ~isfield(opt,'bw')
    % default bandwidth range is 0.1-2
    opt.bw = 0.3:0.1:2;
end
if ~isfield(opt,'C')
    opt.C = 0.05;
end

param = struct('method','CG','ker','rbf', 'arg', 0.5, 'C', opt.C);

N=size(data.X,1);
ns = min(500,N);
did = randsample(N,ns);
data.X=data.X(did,:);
data.y=data.y(did,:);

aset = zeros(length(opt.bw),1);
pset = zeros(length(opt.bw),1);
rset = zeros(length(opt.bw),1);

for j = 1:length(opt.bw)
    param.arg = opt.bw(j);
    model = semisvc(data, param);
    labels = zeros(length(data.y),1);
    labels(model.bsv_ind,1)=1;
    cfm = confusionmat(data.y,labels,'order',[1,0]);
    tp = cfm(1,1); fp = cfm(2,1); fn = cfm(1,2); tn = cfm(2,2);
    [pset(j),rset(j)] = conf2pr(cfm);
    aset(j) = (tp+tn)/(tp+tn+fp+fn);
end
[~, idx] = max(aset);
h = opt.bw(idx);
if(nargout > 1)
    hold on;
    g=gcf;
    plot(opt.bw,aset,'-.ko','DisplayName','Accu');
    plot(opt.bw,pset,'-.rs','DisplayName','prec');
    plot(opt.bw,rset,'-.b^','DisplayName','reca');
end
end

