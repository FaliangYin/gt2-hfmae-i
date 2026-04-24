import matlab.engine
import matlab
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from fmae_image.GT2HFMAE import GT2HFMAE
from fmae_image.FMAE_explainer_np import FMAE_explainer_np
from fmae_image.utilities_blackbox import draw_overlay_heatmap, get_weights, positive_normalize,generate_binary_matrix
from fmae_image.utilities_experiments import input_transform, compress_image
from discussion.utilities_segmentation import superpixel_to_semantic, compute_segment_visibility, segment_to_superpixels
import json
from PIL import Image

'''
Initialization
'''

##### Instance ######
file_name = 'Western_Gull_0087_54193.jpg'
# Western_Gull_0087_54193 Green_Jay_0085_66077
#####################

# load closed box image classifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet121(pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, 200)
model.load_state_dict(torch.load('../classifiers/densenet121_cub200.pth', map_location=device))
model.to(device)
model.eval()

save_path = './result/'
os.makedirs(save_path, exist_ok=True)

eng = matlab.engine.start_matlab()
eng.addpath('../fmae_image/lib_matlab', nargout=0)
eng.rng(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
print('Engine launched')

with open(os.path.abspath('../images/image_net_2012/imagenet_class_index.json'), 'r') as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]


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


'''
Code for Section 2
Experiment analysis: influence of inaccurate semantic segmentation on explanations
'''
instance = np.array(eng.imread(file_name))
instance = compress_image(instance)

explainer = GT2HFMAE(eng)  # initialize the explainer
input_con, output = explainer.explain_instance(instance, batch_predict, semantic_num=2, threshold=[0, 0.1, 0.3, 0.5], show_seg=1,
                           improve=True, require_rules=True, require_dataset=True)  # explain the instance: Fig. 1(a-c), Fig. 2(a-c)
eng.seg_with_idx_max(instance, explainer.semantic_map, 20, 'S', 0, nargout=0)  # show local semantic label (save to an image)
# generated a salience map
# Draw Fig.1(d) Fig.2(d) (saved in result folder)
heatmap = draw_overlay_heatmap(np.array(instance), np.array(explainer.superpixel_map), explainer.fs,
                               save_path=save_path + '/' + file_name + '.png',
                               Label=np.array(explainer.semantic_map), max_label=explainer.max_label, bar=True)

'''
Code for Section 3
Practical solutions with the proposed framework handling inaccurate segmentation
'''
# Segment the instance by pre-trained model DeepLabV3 and merge it with FCM segmentation result
superpixel, Label = np.array(explainer.superpixel_map).astype(int) - 1, np.array(explainer.semantic_map).astype(int) - 1
image = Image.open(file_name).convert("RGB")
# Draw Fig. 3 and Fig. 4 (saved in result folder)
semantic_seg, segment_sizes, sp2seg = superpixel_to_semantic(image, superpixel, Label, file_name=file_name,)
eng.seg_with_idx_max(instance, semantic_seg+1, 20, 'S', 0, nargout=0)  # show result

# train GT2-HFMAE-I explainer based on the refined segmentation
input_ant_enhanced = compute_segment_visibility(semantic_seg, superpixel, sp2seg, input_con)
explainer_enhanced = FMAE_explainer_np(input_ant_enhanced.shape[1], input_con.shape[1], output.shape[1], 2, tr_mode=1)
weights = get_weights(input_con, kernel_width=0.5*np.sqrt(input_ant_enhanced.shape[-1]))
explainer_enhanced.fit(input_ant_enhanced.detach().numpy(), input_con.detach().numpy(), output.detach().numpy(), weights)
fs_enhanced = explainer_enhanced.feature_attribution(input_ant_enhanced)
fs_enhanced = positive_normalize(fs_enhanced)

# Draw Fig. 5 (saved in result folder)
seg2sp = segment_to_superpixels(semantic_seg, superpixel)
membership_enhanced = generate_binary_matrix(seg2sp, np.max(superpixel) + 1)
fs_semantic_ob = np.dot(membership_enhanced, fs_enhanced * explainer.superpixel_size) / np.expand_dims(segment_sizes, axis=1)
enhanced_heatmap = draw_overlay_heatmap(np.array(instance), np.array(superpixel)+1, fs_enhanced,
                                        save_path=save_path+'/'+file_name+'_enhanced'+'.png',
                               Label=semantic_seg, max_label=np.argmax(fs_semantic_ob), bar=True)

pass
