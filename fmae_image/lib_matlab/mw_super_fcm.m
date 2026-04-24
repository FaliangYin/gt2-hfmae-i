function [Lr2,center_Lab,U,iter_n,MG,U1,IDX3]=mw_super_fcm(L1,data, Label_n,cluster_n, S_num, options) %
%% check  input variants
if nargin ~= 5 && nargin ~= 6
    error('Too many or too few input arguments!');
end
data_n = size(data, 1); %the row of input matrix
% Change the following to set default options
default_options = [2;	% exponent for the partition matrix U
    50;	% max. number of iteration
    1e-5;	% min. amount of improvement
    1];	% info display during iteration

if nargin == 5,
    options = default_options;
else
    % If "options" is not fully specified, pad it with default values.
    if length(options) < 5,
        tmp = default_options;
        tmp(1:length(options)) = options;
        options = tmp;
    end
    % If some entries of "options" are nan's, replace them with defaults.
    nan_index = find(isnan(options)==1);
    options(nan_index) = default_options(nan_index);
    if options(1) <= 1,
        error('The exponent should be greater than 1!');
    end
end
expo = options(1);		% Exponent for U
max_iter = options(2);		% Max. iteration
min_impro = options(3);		% Min. improvement
display = options(4);		% Display info or not
iter_n=0; % actual number of iteration
U = initfcm(cluster_n, data_n);			% Initial fuzzy partition
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
iter_n = i;
center_Lab=center;
[~,IDX2]=max(U);
%%
ll=1;rr=0;
for j=1:size(L1,2)   
    rr=rr+S_num(j);
    IDX3{j}=IDX2(ll:rr);
    U1{j}=U(:,ll:rr);
    ll=ll+S_num(j);    
end
%%
Lr2={};
for j=1:size(L1,2)
    Lr21=zeros(size(L1{j},1),size(L1{j},2));
    for i=1:max(L1{j}(:))
        Lr21=Lr21+(L1{j}==i)*IDX3{j}(i);
    end
    Lr2{j}=Lr21;
end
%%
u=0;
for k=1:size(L1,2)
MG{k} = cell(1,cluster_n);
for i=1:cluster_n
    MG{k}{i}=zeros(size(L1{k},1),size(L1{k},2));
    for j=1:max(L1{k}(:))
        MG{k}{i}=MG{k}{i}+(L1{k}==j)*U(i,j+u);
    end
end
u=u+S_num(k);
end