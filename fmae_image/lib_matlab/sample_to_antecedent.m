function [U, clu_act] = sample_to_antecedent(U_, Num_, samples, threshold, cluster, total_semantic)
U=cell(1,length(threshold));
clu_act=cell(1,length(threshold));
for ii=1:length(threshold)
        U_(U_<threshold{ii})=0;
        U{ii} = U_;

        clu_act{ii}=zeros(size(samples,1), cluster);
        for i=1:size(samples, 1)
            IDX_pre=find(samples(i,:)==1);
            for k=1:cluster
                clu_act{ii}(i,k)=sum(U_(k,IDX_pre)*Num_(IDX_pre))/total_semantic(k);
            end
        end
        clu_act{ii}(isnan(clu_act{ii}))=0;
end