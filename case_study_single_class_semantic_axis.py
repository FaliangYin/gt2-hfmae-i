import matlab.engine
import matlab
from torchvision import models
import numpy as np

import torch
from fmae_image.utilities_experiments import input_transform
from fmae_image.utilities_blackbox import (draw_overlay_heatmap, print_rules, generate_binary_matrix)
from fmae_image.GT2HFMAE import GT2HFMAE
from fmae_image.FMAE_explainer import membership_fun
import torch.nn.functional as F
import os


"""
V. A. Case study: single-class explanation task
- Semantic axis
"""

model = models.inception_v3(pretrained=True)  # closed box model
model.eval()
image_folder = './images/image_net_2012/'  # image path
file_name = 'n02102973_1846.JPEG'  # image instance
save_path = './result/single_class/'  # result save path
os.makedirs(save_path, exist_ok=True)

eng = matlab.engine.start_matlab()  # load matlab engine
eng.addpath('./fmae_image/lib_matlab', nargout=0)  # matlab function path
eng.rng(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

def batch_predict(batch):
    # predict samples by black-box model
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = torch.stack([input_transform()(im) for im in batch], dim=0)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

instance = eng.imread(image_folder + file_name)  # load image instance
explainer = GT2HFMAE(eng)  # initialize the explainer
input_con, model_output = explainer.explain_instance(instance, batch_predict, semantic_num=3, threshold=[0, 0.1, 0.3, 0.5],
                                                     show_seg=1, improve=True, require_rules=True,
                                                     require_semantic=True)  # explain the instance
# Draw Fig.1(c) in main paper
eng.seg_with_idx_max(instance, explainer.semantic_map, 16, 'S', 1, save_path + '/' + file_name + '_idx.png', nargout=0)  # show semantic label (save to an image)
y_r = np.expand_dims(explainer.c_score, axis=0)  # classification
# Draw Fig.5(a) in main paper
print_rules(explainer.explainer.rule_base, explainer.explainer.num_fuzzy_set, np.array(y_r),
            save_path + file_name + '.txt')
# Draw Fig.4(c) in main paper
heatmap = draw_overlay_heatmap(np.array(instance), np.array(explainer.superpixel_map), explainer.fs,
                               save_path=save_path + '/' + file_name + '.png',
                               Label=np.array(explainer.semantic_map), max_label=explainer.max_label, bar=True)


# Manually implementing forward propagation for semantic axis case study
fs_semantic_max = np.argmax(explainer.fs_semantic) + 1  # find the most contributive semantic segment
fs_max_idx = np.where((np.array(explainer.semantic_idx)-1).astype(int).squeeze(axis=0)==fs_semantic_max-1)[0].tolist()  # find the superpixels belong to that segment
fs_max = [explainer.fs[i] for i in fs_max_idx]  # record their salience values

# for refined level, divide these superpixels to two parts (eq18): with salience values higher than the average (sub1) and lower (sub2)
sub1, sub2 = [], []
sub1_num, sub2_num = [], []  # their pixel numbers
sub1_mem, sub2_mem = [], []  # their membership degree to their semantic segment
for original_index, value in zip(fs_max_idx, fs_max):
    if value > np.max(explainer.fs_semantic):  # max(explainer.fs_semantic) is the average salience in the most contributive semantic segment
        sub1.append(original_index)
        sub1_num.append(explainer.superpixel_size[original_index][0])
        sub1_mem.append(explainer.membership[int(fs_semantic_max - 1)][original_index])
    else:
        sub2.append(original_index)
        sub2_num.append(explainer.superpixel_size[original_index][0])
        sub2_mem.append(explainer.membership[int(fs_semantic_max - 1)][original_index])

# for simplified level, divide all the superpixels to two parts (eq17): contribute to the result(subo), and not (subb)
fs_bkg_idx = np.where((np.array(explainer.semantic_idx) - 1).astype(int).squeeze(axis=0) != fs_semantic_max - 1)[0].tolist()
semantic_idx_ = (np.array(explainer.semantic_idx) - 1).astype(int).squeeze(axis=0).tolist()
subb, subb_num, subb_mem, = [], [], []
for original_index in fs_bkg_idx:
    subb.append(original_index)
    subb_num.append(explainer.superpixel_size[original_index][0])
    subb_mem.append(explainer.membership[semantic_idx_[original_index]][original_index])
subo, subo_num, subo_mem, = [], [], []
for original_index in fs_max_idx:
    subo.append(original_index)
    subo_num.append(explainer.superpixel_size[original_index][0])
    subo_mem.append(explainer.membership[semantic_idx_[original_index]][original_index])

# Eq.5 (input_con[:, sub1] is Eq. 6)
wei1 = np.sum(input_con[:, sub1] * np.array(sub1_num), axis=1) / np.sum(np.array(sub1_num))
wei2 = np.sum(input_con[:, sub2] * np.array(sub2_num), axis=1) / np.sum(np.array(sub2_num))
weib = np.sum(input_con[:, subb] * np.array(subb_num), axis=1) / np.sum(np.array(subb_num))
weio = np.sum(input_con[:, subo] * np.array(subo_num), axis=1) / np.sum(np.array(subo_num))
# Eq. 8 for refined level
wei_mem = membership_fun(torch.Tensor(np.column_stack((wei1, wei2, weib))).unsqueeze(1),
                         torch.Tensor([[0, 0, 0], [1, 1, 1]]))
wei_bar = wei_mem[:, explainer.explainer.rule_base, range(3)].prod(dim=2)
wei_bar_bar = wei_bar / torch.sum(wei_bar, 0, keepdim=True)
# Eq. 8 for simplified level
weio_mem = membership_fun(torch.Tensor(np.column_stack((weio, weib))).unsqueeze(1),
                          torch.Tensor([[0, 0], [1, 1]]))
weio_bar = weio_mem[:, torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).int(), range(2)].prod(dim=2)
weio_bar_bar = weio_bar / torch.sum(weio_bar, 0, keepdim=True)
# Eq. 15
member_ob = generate_binary_matrix([subo, subb], input_con.shape[1])
member_s = generate_binary_matrix([sub1, sub2, subb], input_con.shape[1])
fs_semantic_ob = np.dot(member_ob, explainer.fs * explainer.superpixel_size[0]) / np.expand_dims([np.sum(subo_num), np.sum(subb_num)], axis=1)
fs_semantic_s = np.dot(member_s, explainer.fs * explainer.superpixel_size[0]) / np.expand_dims(
    [np.sum(sub1_num), np.sum(sub2_num), np.sum(subb_num)], axis=1)
# Eq. 16
y_r_s = torch.einsum('MC,MR->CR', model_output.float(), wei_bar_bar).data.numpy()
y_r_ob = torch.einsum('MC,MR->CR', model_output.float(), weio_bar_bar).data.numpy()

# Draw Fig.5(b) in main paper for simplified level
print_rules(torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).int(), explainer.explainer.num_fuzzy_set, np.array(y_r_ob),
            save_path + file_name + '_simplified.txt')
# Draw Fig.5(c) in main paper for refined level
print_rules(explainer.explainer.rule_base, explainer.explainer.num_fuzzy_set, np.array(y_r_s),
            save_path + file_name + '_refined.txt')

pass