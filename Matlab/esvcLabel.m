function [model] = esvcLabel(data,model)

%==========================================================================
%  E-SVC Labeling
%  
%  Reference
%  J. Lee and D. Lee, Dynamic Characterization of Cluster Structures for 
%  Robust and Inductive Support Vector Clustering, IEEE TPAMI 28-(11), 
%  1869-1874, 2006.
%
%==========================================================================
% January 13, 2009
% Implemented by Daewon Lee
% WWW: http://sites.google.com/site/daewonlee/
%==========================================================================

%
X=data.X';
y=data.y';
K=model.options.NofK;

[N, attr] = size(X);

% Find SEPs
[rep_locals,locals,local_val,match_local]=FindLocal(X,model);
NofLocals=size(locals,1);
model.local=locals';

% Find transition points and label the SEPs
[ts]=Find_TSs(locals,model);   
nOfTS=length(ts.f);

%% Cluster assignment of each data point

% --- Automatic determination of cluster number based on the cluster boundary
if K==0 
    disp('Automatic determination of cluster numbers based on the SVDD boundearies defined by R^2');
    adjacent = zeros(NofLocals,NofLocals);
    tmp=find(ts.f<model.r+10^(-7));
    
    if ~isempty(tmp)    % only check the connectivity of TSs inside the sphere
        for j=1:length(tmp)
            adjacent(ts.neighbor(tmp(j),1),ts.neighbor(tmp(j),2))=1;
            adjacent(ts.neighbor(tmp(j),2),ts.neighbor(tmp(j),1))=1;
        end
        %% To connect nodes which can be connected via directly connected edges.
        for i=1:NofLocals
            for j=1:i
                if (adjacent(i,j) == 1)
                    adjacent(i,:) = (adjacent(i,:) | adjacent(j,:));
                end
            end
        end
        local_clusters_assignments = FindConnectedComponents(adjacent);
    end  
    
    % model update
    model.ts.x=ts.x(tmp,:);
    model.ts.f=ts.f(tmp,:);
    model.ts.purturb=ts.purturb(tmp,:);
    model.ts.neighbor=ts.neighbor(tmp,:);
    model.ts.cuttingLevel=model.r;
    
    % cluster assignment into entire data points
    model.cluster_labels = local_clusters_assignments(match_local)';

% --- Controling the number of clusters 
else    
    local_clusters_assignments=[];
    %ts.f
    %model.rsq
    f_sort=sort(ts.f);
    
    adjacent = zeros(NofLocals,NofLocals,nOfTS);
    a=[];
    flag=0;
    for m=1:nOfTS
        cur_f=f_sort(end+1-m);    % cutting level:large --> small  (small number of clusters --> large number of clusters)
        %cur_f=f_sort(i);         % cutting level: small --> large (large number of clusters --> small number of clusters)
        tmp=find(ts.f<cur_f);
        if ~isempty(tmp) % No TSs inside the sphere
            for j=1:length(tmp)
                adjacent(ts.neighbor(tmp(j),1),ts.neighbor(tmp(j),2),m)=1;
                adjacent(ts.neighbor(tmp(j),2),ts.neighbor(tmp(j),1),m)=1;
            end
            %% To connect nodes which can be connected via directly connected edges.
            for i=1:NofLocals
                for j=1:i
                    if (adjacent(i,j,m) == 1)
                        adjacent(i,:,m) = (adjacent(i,:,m) | adjacent(j,:,m));
                    end
                end
            end            
            
        end % end of current TS
        a=[a;cur_f];
        my_ts.x=ts.x(tmp,:);
        my_ts.f=ts.f(tmp,:);
        my_ts.purturb=ts.purturb(tmp,:);
        my_ts.neighbor=ts.neighbor(tmp,:);
        my_ts.cuttingLevel=cur_f;
        ind=find(ts.f==cur_f);
        my_ts.levelx=ts.x(ind(1),:);
        tmp_ts{m}=my_ts;
        
        assignment = FindConnectedComponents(adjacent(:,:,m));
        
        if max(assignment)==K
            disp('We can find the number of K clusters');         
            % model update
            model.ts=tmp_ts{m};    
            % cluster assignment into entire data points
            model.cluster_labels =  assignment(match_local)';       
            flag=1;
            break;
        end
        local_clusters_assignments = [local_clusters_assignments assignment];                        
    end % end of K-control
    
    % cannot find k clusters     
    if flag==0
        disp('Cannot find cluster assignments with K number of clusters, instead that we find cluster assignmentsthe with the nearest number of clusters to K !');
        [dummy,ind]=min(dist2(max(local_clusters_assignments)',K));
        
        %ts=[];
        model.ts=tmp_ts{ind(1)};
        local_clusters_assignments=local_clusters_assignments(:,ind(1));
        model.cluster_labels = local_clusters_assignments(match_local);
    end
end



