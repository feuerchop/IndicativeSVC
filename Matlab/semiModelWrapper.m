function res = semiModelWrapper( data, model, params)
%SEMIMODELWRAPPER Summary of this function goes here
%   Semi-supervised model wrapper
% INPUT:
%   data.X: N*d data set
%   data.y: Test labels
%   model: Name of the model
%   params: models encoded parameters
switch model
    case 'svc'
        % Normla SVC algorithm without given labels
        params.Xa = [];
        params.Xr = [];     
        % init opt
        model=semisvc(data,params);
        % predicted labels
        labels = zeros(length(data.y),1);
        labels(model.bsv_ind,1)=1;
    case 'isvc'
        % init model options
        if isempty(params.Xa) && isempty(params.Xr)
            error('Should give labels to iSVC algorithm, exit!');
        end
        model=semisvc(data,params);
        % predicted labels
        labels = zeros(length(data.y),1);
        labels(model.bsv_ind,1)=1;
    case 's3vm'
        % init model options        
        if ~isfield(params,{'Xa'})
            params.Xa=[];
        end
        if ~isfield(params,{'Xr'})
            params.Xr=[];
        end
        svmlin_labels = zeros(length(data.y),1);
        svmlin_labels(params.Xa)=1;
        svmlin_labels(params.Xr)=-1;
        svmlin_path='~/Documents/MATLAB/svmlin/';
        svmlinwrite([svmlin_path,'testdata'],data.X,svmlin_labels);
        labels=svmlinGo('testdata',params);
    case 'lds'
        if ~isfield(params,{'Xa'})
            params.Xa=[];
        end
        if ~isfield(params,{'Xr'})
            params.Xr=[];
        end
        
    otherwise
        error('No such algorithm is supported, exit!');
end
% process results
res.cfm = confusionmat(data.y,labels,'order',[1,0]);
[res.prec,res.reca,res.acc,res.f1,res.fpr,res.fnr] = conf2pr(res.cfm);
end

