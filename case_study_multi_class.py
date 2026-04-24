import matlab.engine
import matlab
from torchvision import models
import numpy as np

import torch
from fmae_image.utilities_experiments import input_transform
from fmae_image.utilities_blackbox import (draw_overlay_heatmap, print_rules)
from fmae_image.GT2HFMAE import GT2HFMAE
import torch.nn.functional as F
import os

"""
S.I. EXTENSION OF CASE STUDY: MULTI-CLASS EXPLANATION TASK
"""

# Task initialization
class_list = [0, 1, 2]  # indices of class to be explained
model = models.inception_v3(pretrained=True)  # closed box model
model.eval()
image_folder = './images/image_net_2012/'  # image path
file_name = 'n02094114_3253.JPEG'  # image instance
save_path = './result/multi_class/'  # result save path
os.makedirs(save_path, exist_ok=True)

eng = matlab.engine.start_matlab()  # load matlab engine
eng.addpath('./fmae_image/lib_matlab', nargout=0)  # matlab function path
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

instance = eng.imread(image_folder + file_name)  # load image instance
explainer = GT2HFMAE(eng)  # initialize the explainer
explainer.explain_instance(instance, batch_predict, semantic_num=3, threshold=[0, 0.1, 0.3, 0.5], show_seg=1, improve=True,
                           require_rules=True, class_list=class_list)  # explain the instance

# Draw Fig.2(a) in Supplementary Materials
eng.seg_with_idx_max(instance, explainer.semantic_map, 16, 'S', 1, save_path + '/' + file_name + '_idx.png', nargout=0)  # show semantic label (save to an image)
# Draw Fig.2(e) in Supplementary Materials
print_rules(explainer.explainer.rule_base, explainer.explainer.num_fuzzy_set, np.array(explainer.c_score),
            save_path + file_name + '.txt', is_print=True)  # save the rules

for explained_class in class_list:  # for each explained class, generated a salience map
    # Draw Fig.2(b)(c)(d) in Supplementary Materials
    heatmap = draw_overlay_heatmap(np.array(instance), np.array(explainer.superpixel_map), explainer.fs[:,explained_class],
                                   save_path=save_path + '/' + file_name+ '_class' + str(explained_class) + '.png',
                                   Label=np.array(explainer.semantic_map), max_label=np.argmax(explainer.fs_semantic[:,explained_class]) + 1, bar=True)
pass