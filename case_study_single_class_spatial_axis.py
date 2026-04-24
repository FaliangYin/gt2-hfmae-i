import matlab.engine
import matlab
from torchvision import models
import numpy as np
import torch
from fmae_image.utilities_experiments import input_transform
from fmae_image.utilities_blackbox import (draw_overlay_heatmap, binary_matrix_to_decimal_vector,
                                           print_rules, positive_normalize, single_bar)
from fmae_image.GT2HFMAE import GT2HFMAE
import torch.nn.functional as F
import os


"""
V. A. Case study: single-class explanation task
- Spatial axis
"""

# Task initialization
model = models.inception_v3(pretrained=True)  # closed box model
model.eval()
image_folder = './images/image_net_2012/'
file_names = ['n02102973_1846.JPEG', 'n02102973_4328.JPEG', 'n02102973_721.JPEG']  # image instances
save_path = './result/single_class/'  # result save path
os.makedirs(save_path, exist_ok=True)

eng = matlab.engine.start_matlab()
eng.addpath('./fmae_image/lib_matlab', nargout=0)
eng.rng(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


def batch_predict(batch):
    # predict samples by closed box model
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = torch.stack([input_transform()(im) for im in batch], dim=0)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


instances, semantic_numbers, centerLabs, S_num, Label, FRBs, c_scores, fs_semantics, fss = [], [], [], [], [], [], [], [], []
for i, file_name in enumerate(file_names):  # explain each image instance
    instances.append(eng.imread(image_folder + file_name))  # load image instance
    explainer = GT2HFMAE(eng)  # initialize the explainer
    explainer.explain_instance(instances[i], batch_predict, semantic_num=3, threshold=[0, 0.1, 0.3, 0.5], show_seg=1,
                               improve=True, require_rules=True, require_spatial=True)  # explain the instance
    # eng.seg_with_idx_max(instances[i], explainer.semantic_map, 16, 'S', 1, save_path + '/' + file_name + '_idx.png', nargout=0)  # show local semantic label (save to an image)
    # generated a salience map
    # Draw Fig.4(c)(g)(h) in main paper
    heatmap = draw_overlay_heatmap(np.array(instances[i]), np.array(explainer.superpixel_map), explainer.fs,
                                   save_path=save_path + '/' + file_name + '.png',
                                   Label=np.array(explainer.semantic_map), max_label=explainer.max_label, bar=True)
    # record the parameters for spatial-axis hierarchical explanations
    fs_semantics.append(np.squeeze(explainer.fs_semantic, axis=1))
    c_scores.append(np.expand_dims(explainer.c_score, axis=0))
    centerLabs.append(explainer.center_lab)
    semantic_numbers.append(explainer.semantic_size)
    S_num.append(explainer.semantic_num_act)
    Label.append(explainer.semantic_map)
    FRBs.append(np.array(explainer.explainer.rule_base))

# align the semantic features across instances in the domain
domain_semantic_map, IDXs = eng.semantic_fcm(centerLabs, semantic_numbers, 3, matlab.double(S_num), Label, nargout=2)  # align the features
for i, IDX in enumerate(IDXs):
    # Draw Fig.1(c) Fig.5(e)(f) in main paper
    eng.seg_with_idx_max(instances[i], domain_semantic_map[i], 16, 'S', 1, save_path + '/' + file_names[i] + '_idx_new.png', nargout=0)  # show domain semantic label (save to an image)
    IDX = np.array(IDX).squeeze(axis=0).astype(int)-1  # map local feature index to domain feature index
    FRBs_ = FRBs[i][:, IDX.tolist()]
    fs_semantics[i] = fs_semantics[i][IDX.tolist()]
    new_id = binary_matrix_to_decimal_vector(FRBs_)
    y_r = c_scores[i][:, new_id]
    # Draw Fig.5(a) in main paper
    print_rules(FRBs[i], 2, y_r, save_path + 'case1_' + file_names[i] + '.txt')
    c_scores[i] = y_r

fs_semantics = np.array(fs_semantics)
normalized_fs_semantics = positive_normalize(fs_semantics, axis=1, mode='norm')
file_names = [ii[:-5] for ii in file_names]
# Draw Fig.4(d) in main paper
single_bar(normalized_fs_semantics, save_path=save_path)
y_rs_mean = np.mean(c_scores, axis=0)
y_rs_std = np.std(c_scores, axis=0)
# Draw Fig.5(d) in main paper
print_rules(FRBs[0], 2, y_rs_mean, save_path + 'case1_domain.txt', std=y_rs_std)
pass