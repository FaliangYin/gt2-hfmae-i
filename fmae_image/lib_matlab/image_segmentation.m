function [superpixel_map, semantic_map, superpixel_size, semantic_size, membership, semantic_sum, IDX2, centerLab2] = image_segmentation(f_ori, cluster_num, show_plt, save_plt, save_name)
L1=w_MMGR_WT(f_ori,3);
superpixel_map=imdilate(L1,strel('square',2));
[plot1,~,superpixel_size,centerLab1]=Label_image(f_ori,superpixel_map);
[semantic_map,~,membership,~,MG,IDX2]=w_super_fcm(superpixel_map,centerLab1,superpixel_size,cluster_num);
[plot2,~,semantic_size,centerLab2]=Label_image(f_ori,semantic_map);

BW = boundarymask(superpixel_map);
plot1=imoverlay(plot1,BW,'cyan');
BW = boundarymask(semantic_map);
plot2=imoverlay(plot2,BW,'cyan');

if show_plt==1
    f = figure;
    tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
    
    nexttile, imshow(f_ori, 'InitialMagnification', 'fit'); axis off
    nexttile, imshow(plot1, 'InitialMagnification', 'fit'); axis off
    nexttile, imshow(plot2, 'InitialMagnification', 'fit'); axis off

    if save_plt==1
        exportgraphics(f, save_name, 'Resolution', 300)
    end
end

semantic_sum=zeros(1,cluster_num);
for k=1:cluster_num
    semantic_sum(k) = sum(sum(MG{k}));
end
    
end