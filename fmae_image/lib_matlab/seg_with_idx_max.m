function seg_with_idx_max(img,segments,FontSize, title, save_flag, save_name)
labeled_img=Label_image(img,segments);
BW = boundarymask(segments);
highlightedImage = imoverlay(labeled_img, BW,'cyan');
figure
imshow(highlightedImage);
hold on;

regions = unique(segments);
regions(regions == 0) = []; % 去除背景（假设背景索引为0）

% 扩展分割边缘图像，以便计算距离时考虑边界
extendedBW = BW;
extendedBW([1:2, end-1:end], :) = true;
extendedBW(:, [1:2, end-1:end]) = true;
for k = 1:length(regions)
    regionIndex = regions(k);
    
    % 提取当前索引的子图像
    subImage = (segments == regionIndex);
    
    % 找到该索引区域的所有连通组件
    CC = bwconncomp(subImage);

    % 获取所有连通组件的区域性质
    stats = regionprops(CC, 'Area', 'PixelIdxList');

    % 找到面积最大的连通组件
    [~, maxIdx] = max([stats.Area]);
    maxPixelIdxList = stats(maxIdx).PixelIdxList;

    % 转换为子图像的行列坐标
    [rows, cols] = ind2sub(size(subImage), maxPixelIdxList);

    % 计算分割边缘的距离变换
    D = bwdist(~subImage | extendedBW);
   
    % 计算当前连通组件的边界框
    singleComponent = false(size(subImage));
    singleComponent(maxPixelIdxList) = true;
    boundingBox = regionprops(singleComponent, 'BoundingBox');
    
    % 获取边界框的宽度和高度
    bboxWidth = boundingBox.BoundingBox(3);
    bboxHeight = boundingBox.BoundingBox(4);
    minDist = round(0.1 * min(bboxWidth, bboxHeight)); % 例如，取10%
    
    % 找到距边缘至少 minDist 的点
    validIdx = find(D(maxPixelIdxList) >= minDist);
    
    if isempty(validIdx)
        % 如果没有找到满足条件的点，使用距离最大的点
        [~, idx] = max(D(maxPixelIdxList));
        centerRow = rows(idx);
        centerCol = cols(idx);
    else
        % 从符合条件的点中选择距离最大的点
        [~, idx] = max(D(maxPixelIdxList(validIdx)));
        centerRow = rows(validIdx(idx));
        centerCol = cols(validIdx(idx));
    end
    
    % 在连通组件的中心位置标记索引数字
    if iscell(title)
        text(centerCol, centerRow, title{k}, 'Color', 'magenta', 'FontSize', FontSize, 'FontWeight', 'bold');
    else
        text(centerCol, centerRow, [title num2str(regionIndex)], 'Color', 'magenta', 'FontSize', FontSize, 'FontWeight', 'bold');
    end
    
    % % 遍历每个连通组件
    % for j = 1:CC.NumObjects
    %     % 获取当前连通组件的像素索引
    %     pixelIdxList = CC.PixelIdxList{j};      
    % end
end
hold off;
if save_flag == 1
    % saveas(gcf, save_name);
    exportgraphics(gcf, save_name, 'ContentType', 'image')
end
end

