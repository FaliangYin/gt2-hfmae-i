function [mask_imgs] = image_exp_inv_1(segments,images_path,S_num)
image_files = dir(fullfile(images_path, '*.JPEG'));
mask_imgs=cell(1,S_num);
for j=1:S_num
    mask_img = uint8(zeros([size(segments,1), size(segments,2), 3]));
    randomIndex = randi(length(image_files),1,max(max(segments)));
    for i=1:max(max(segments))        
        filename = fullfile(images_path, image_files(randomIndex(i)).name);
        f_ori = imread(filename);
        f_ori = imresize(f_ori, [size(segments,1), size(segments,2)]);    
        mask_img(repmat(segments == i, [1, 1, 3])) = f_ori(repmat(segments == i, [1, 1, 3]));
    end
    mask_imgs{j}=mask_img;
end
end
