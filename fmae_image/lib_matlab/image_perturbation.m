function [samples_image] = image_perturbation(f_ori_, L2_, N_sam, samples)
%%
%samples=randi([0,1],[N_sam,size(Num_,1)]);
%samples(1,:)=1;
%%
samples_image=cell(1,N_sam);
for i=1:N_sam        
    IDX_abs=find(samples(i,:)==0);
    samples_image{i}=f_ori_;
    samples_image{i}(repmat(ismember(L2_, IDX_abs), [1, 1, size(f_ori_, 3)])) = 255;
end
    

