function [Lr2, IDX3] = semantic_fcm(data, Label_n, cluster_n, S_num, Labels)
if iscell(data)
    data=cell2mat(data');
    Label_n=cell2mat(Label_n');
end
expo = 2;	% exponent for the partition matrix U
max_iter = 50;		% max. number of iteration
min_impro = 1e-5;		% min. amount of improvement
data_n = size(data, 1);
U = initfcm(cluster_n, data_n);			% Initial fuzzy partition
% U=rand(cluster_n, data_n);
% U = U ./ sum(U);
Num=ones(cluster_n,1)*Label_n';
for i = 1:max_iter
    mf = Num.*U.^expo;       % MF matrix after exponential modification
    center = mf*data./((ones(size(data, 2), 1)*sum(mf'))'); % new center
    out = zeros(size(center, 1), size(data, 1));
    if size(center, 2) > 1
        for k = 1:size(center, 1)
            out(k, :) = sqrt(sum(((data-ones(size(data, 1), 1)*center(k, :)).^2)'));
        end
    else	% 1-D data
        for k = 1:size(center, 1)
            out(k, :) = abs(center(k)-data)';
        end
    end
    dist=out+eps;
    tmp = dist.^(-2/(expo-1));
    U = tmp./(ones(cluster_n, 1)*sum(tmp)+eps);
    Uc{i}=U;
    if i> 1
        if abs(max(max(Uc{i} - Uc{i-1}))) < min_impro, break; end
    end
end
center_Lab=center;
[~,IDX2]=max(U);
%%
%%
IDX3={};
ll=1;rr=0;
for j=1:size(S_num,2)   
    rr=rr+S_num(j);
    IDX3{j}=IDX2(ll:rr);
    ll=ll+S_num(j);    
end
%%
Lr2={};
for j=1:size(Labels,2)
    Lr21=zeros(size(Labels{j},1),size(Labels{j},2));
    for i=1:max(Labels{j}(:))
        Lr21=Lr21+(Labels{j}==i)*IDX3{j}(i);
    end
    Lr2{j}=Lr21;
end
end

