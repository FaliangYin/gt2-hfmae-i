import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
from fmae_image.utilities_experiments import compress_image
from fmae_image.utilities_blackbox import generate_binary_matrix
from skimage import transform


def merge_segmentations(seg1, seg2):
    # merge two semantic segmentation results
    assert seg1.shape == seg2.shape, "unmatched size of seg1 and seg2"
    max_id_seg1 = seg1.max()
    seg2_shifted = seg2.copy()
    # shift the indices of seg2 except background 0
    mask_nonbg = seg2 > 0
    seg2_shifted[mask_nonbg] = seg2_shifted[mask_nonbg] + max_id_seg1
    # overlay seg2 (without background) to seg1
    merged = seg1.copy()
    merged[mask_nonbg] = seg2_shifted[mask_nonbg]
    return merged


def relabel_contiguous(seg):
    # Check whether the indexes in the index matrix are continuous. If not, renumber them as continuous integers.
    # Ensure that the sequence remains unchanged starting from 0.
    unique_ids = np.unique(seg)
    mapping = {old: new for new, old in enumerate(unique_ids)}  # mapping dict: new to old
    new_seg = np.vectorize(mapping.get)(seg)
    return new_seg, mapping


def superpixel_to_semantic(image, superpixel, base_semantic=None, file_name=''):
    # generate a new segmentation by DeepLabV3 and merge it with the base segmentation from FCM
    plt.figure(figsize=(8, 4))
    plt.subplot(231)
    plt.imshow(image)
    plt.title("Original image")
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(superpixel, cmap='tab20')
    plt.title("Superpixels")
    plt.axis('off')

    if base_semantic is not None:
        plt.subplot(233)
        plt.imshow(base_semantic, cmap='tab20')
        plt.imsave(f'./result/{file_name}_1.png', base_semantic, cmap='tab20')
        plt.title("Base segmentation")
        plt.axis('off')

    # load pre-trained DeepLabV3
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(superpixel.shape),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(image).unsqueeze(0) # [1, 3, H, W]
    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    probs = torch.softmax(output, dim=0)  # [C, H, W]
    probs = probs.cpu().numpy()
    pred_semantic = output.argmax(0).cpu().numpy()

    plt.subplot(234)
    plt.imshow(pred_semantic, cmap='tab20')
    plt.imsave(f'./result/{file_name}_2.png', pred_semantic, cmap='tab20')
    plt.title("Assistant segmentation")
    plt.axis('off')

    pixel_probs = probs.max(axis=0)  # [H, W] probability of each pixel
    pixel_labels = probs.argmax(axis=0)  # [H, W] label of each pixel

    # match the semantic segmentation with superpixel segmentation
    superpixel_labels = []
    superpixel_confidences = {}
    superpixel_sizes = {}  # 每个超像素的像素个数

    for sp_id in np.unique(superpixel):
        mask = (superpixel == sp_id)  # bool mask
        sp_pixel_labels = pixel_labels[mask]
        sp_pixel_probs = pixel_probs[mask]
        labels_in_sp = pred_semantic[mask]
        sp_size = np.sum(mask)

        # voting method
        sp_label = Counter(labels_in_sp).most_common(1)[0][0]

        # Confidence level: the average of the max probabilities of the pixels corresponding to this label
        sp_confidence = sp_pixel_probs[sp_pixel_labels == sp_label].mean()

        superpixel_labels.append(sp_label)
        superpixel_confidences[sp_id] = sp_confidence
        superpixel_sizes[sp_id] = sp_size

    # obtain the new segmentation
    semantic_seg = np.zeros_like(superpixel, dtype=np.int32)
    for sp_id, label in enumerate(superpixel_labels):
        semantic_seg[superpixel == sp_id] = label

    plt.subplot(235)
    plt.imshow(semantic_seg, cmap='tab20')
    plt.imsave(f'./result/{file_name}_3.png', semantic_seg, cmap='tab20')
    plt.title("Superpixel-matched\n assistant segmentation")
    plt.axis('off')

    if base_semantic is None:
        segment_sizes = []
        for label in np.unique(semantic_seg):
            segment_sizes.append(np.sum(semantic_seg == label))
        plt.tight_layout()
        plt.show()
        return semantic_seg, segment_sizes, superpixel_labels
    else:
        # merge the base segmentation with the new segmentation
        merged_semantic = merge_segmentations(base_semantic, semantic_seg)
        merged_semantic, _ = relabel_contiguous(merged_semantic)
        segment_sizes = []
        for label in np.unique(merged_semantic):
            segment_sizes.append(np.sum(merged_semantic == label))
        merged_superpixel_labels = []
        for sp_id in np.unique(superpixel):
            mask = (superpixel == sp_id)
            sp_label = Counter(merged_semantic[mask]).most_common(1)[0][0]
            merged_superpixel_labels.append(sp_label)
        plt.subplot(236)
        plt.imshow(merged_semantic, cmap='tab20')
        plt.imsave(f'./result/{file_name}_4.png', merged_semantic, cmap='tab20')
        plt.title("Merged segmentation")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        return merged_semantic, segment_sizes, merged_superpixel_labels


def compute_segment_visibility(semantic_seg, superpixels, sp2seg, samples):
    # segmentation to PMF
    H, W = semantic_seg.shape
    N = np.max(superpixels) + 1  # superpixel number
    M = samples.shape[0]
    sp_sizes = np.bincount(superpixels.ravel(), minlength=N)
    # find unique semantic labels
    seg_labels = np.unique(semantic_seg)
    seg2idx = {seg: i for i, seg in enumerate(seg_labels)}
    K = len(seg_labels)
    # assign superpixels to their corresponding semantic segment (N, K)
    assign = np.zeros((N, K), dtype=int)
    for sp_id in range(N):
        seg_id = sp2seg[sp_id]
        assign[sp_id, seg2idx[seg_id]] = sp_sizes[sp_id]
    # the present pixel number of each segment in samples
    selected_pixels = samples @ assign  # (M, K)
    # total pixel number in each segment
    seg_totals = np.sum(assign, axis=0)  # (K,)
    # the ratio as PMF
    ratios = selected_pixels / seg_totals[np.newaxis, :]
    return ratios


def segment_to_superpixels(semantic_seg, superpixels):
    # Find the set of superpixels contained in each semantic segment
    seg2sp = []
    for seg_id in np.unique(semantic_seg):
        mask = (semantic_seg == seg_id)
        sp_ids = np.unique(superpixels[mask])
        # seg2sp[seg_id] = set(sp_ids.tolist())
        seg2sp.append(sp_ids.tolist())
    return seg2sp # (dict): {semantic segment ID: set of superpixel ID}
