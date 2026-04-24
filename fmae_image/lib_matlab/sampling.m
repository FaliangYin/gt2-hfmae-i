function [samples] = sampling(Nums_,N_sam)
samples=randi([0,1],[N_sam,size(Nums_,1)]);
samples(1,:)=1;
end

