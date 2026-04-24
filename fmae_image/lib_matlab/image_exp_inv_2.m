function [exp_out, inv_out] = image_exp_inv_2(f_ori_,mask, mask_for_inv,mask_imgs)
S_num=size(mask_imgs,2);
exp_out=cell(1,S_num); inv_out=cell(1,S_num);
for j=1:S_num
    mask_img = mask_imgs{j};    
    exp=uint8(double(f_ori_).*mask+double(mask_img).*~mask);
    inv=uint8(double(f_ori_).*~mask+double(mask_img).*mask_for_inv);
    exp_out{j}=exp;
    inv_out{j}=inv;
end
if S_num==1
    exp_out=exp;
    inv_out=inv;
end
end
