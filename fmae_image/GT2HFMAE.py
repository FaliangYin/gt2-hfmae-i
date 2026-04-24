import numpy as np
import torch
from fmae_image.FMAE_explainer import FMAE_explainer, train_full_batch
from fmae_image.FMAE_explainer_np import FMAE_explainer_np

from fmae_image.utilities_blackbox import get_predicted_label, positive_normalize, remove_small_regions
from scipy.ndimage import binary_fill_holes


class GT2HFMAE:
    def __init__(self, eng):
        super().__init__()
        self.eng = eng  # matlab engine
        self.superpixel_map = None  # map of superpixel segmentation
        self.semantic_map = None  # map of semantic segmentation
        self.superpixel_size = None  # pixel number of each superpixel
        self.semantic_size = None  # pixel number of each semantic segment
        self.center_lab = None  # center color for semantic segments
        self.semantic_num = None  # number of semantic feature (semantic segment)
        self.semantic_num_act = None # actual number of semantic feature
        self.semantic_idx = None  # semantic segment index of each superpixel
        self.membership = None  # membership degree of superpixels to semantic segments
        self.c_score = None  # classification scores
        self.fs = None  # feature salience value of each superpixel feature
        self.fs_semantic = None  # feature salience value of each semantic feature
        self.max_label = None  # the semantic segment contributing to the classification result
        self.score = None  # accuracy of explainer
        self.explainer = None  # GT2FLS explainer

        # interpretability indicators
        self.exp_fea_num = None  # number of features in explanation
        self.num_rules = None  # number of rules in explanation
        self.num_con_para = None  # number of parameters in cosequent part of rules

    def explain_instance(self, instance, batch_predict_fn, semantic_num, threshold, num_fuzzy_set=2,
                         sample_num=2000, batch_size=10, show_seg=0, class_idx=0, improve=False, require_dataset = False,
                         require_rules=False, require_spatial=False, require_semantic=False, class_list = None, training_mode='torch', tr_mode=2):
        '''
        :param instance: instance of interest
        :param batch_predict_fn: closed box model prediction function
        :param semantic_num: number of semantic features (semantic segment)
        :param threshold: threshold epsilon
        :param sample_num: sample number
        :param batch_size: batch size of sample classification
        :param show_seg: show image segmentation result or not (1 or 0)
        :param class_idx: top which class to explain (default: class with max probability)
        :param improve: improve the explanation segment (fill holes and remove isolated regions) or not
        :param require_rules: need output rules?
        :param require_spatial: need hierarchical explanations on spatial axis?
        :param require_semantic: need hierarchical explanations on semantic axis?
        :param require_dataset: need dataset which trained the FLS?
        :param class_list: list of top K class to explain, e.g. [0, 1, 2] -> explain top 3 classes
        '''
        self.semantic_num = semantic_num
        superpixel_map, semantic_map, superpixel_size, semantic_size, membership, semantic_sum, semantic_idx, center_lab = \
            self.eng.image_segmentation(instance, semantic_num, show_seg, 0, nargout=8)  # image segmentation  _mat
        ## semantic_sum is the denominator of PMF for each semantic segment
        # generate perturbation samples in binary representation (the input in superpixel space)
        input_con = self.eng.sampling(superpixel_size, sample_num)
        self.superpixel_map, self.semantic_map = np.array(superpixel_map), np.array(semantic_map)
        self.superpixel_size = superpixel_size; self.semantic_size = semantic_size
        if require_semantic:  # transfer the parameters for simplified or refined level explanation
            self.semantic_idx = semantic_idx; self.membership = membership
        if require_spatial: # transfer the parameters for domain or universe level explanation
            self.center_lab = center_lab; self.semantic_num_act = semantic_size.size[0]

        # classify the perturbation samples by the closed box model
        output = []
        for j in range(0, len(input_con), batch_size):
            # generate perturbation samples in image representation
            samples_image = np.array(self.eng.image_perturbation(instance, superpixel_map, batch_size,
                                                                 input_con[j:j + batch_size]))
            predictions_ = batch_predict_fn(samples_image)
            output.extend(predictions_)
        if class_list is None:
            output, _ = get_predicted_label(output, class_idx+1)
            if class_idx != 0:
                output = np.expand_dims(output[:, class_idx], axis=1)
        else:
            output, _ = get_predicted_label(output, len(class_list))
        # calculate the input of antecedent (input in semantic space)
        membership_semantic, input_ant = self.eng.sample_to_antecedent(membership, superpixel_size, input_con, threshold,
                                                                       self.semantic_num, semantic_sum, nargout=2)
        input_ant, input_con = np.array(input_ant), np.array(input_con)

        self.exp_fea_num = self.semantic_num
        self.num_rules = num_fuzzy_set ** self.exp_fea_num
        self.num_con_para = output.shape[1] * self.num_rules * (superpixel_size.size[0] + 1)

        # train an GT2FLS as the surrogate model
        if training_mode != 'LSE':
            self.explainer = FMAE_explainer(self.semantic_num, superpixel_size.size[0], output.shape[1], 2, tr_mode=tr_mode,
                                            threshold_num=len(threshold))
            input_ant, input_con, output = torch.tensor(input_ant), torch.tensor(input_con), torch.tensor(output)
            train_full_batch(input_ant, input_con, self.explainer, output)
        else:
            self.explainer = FMAE_explainer_np(self.semantic_num, superpixel_size.size[0], output.shape[1], 2,
                                  tr_mode=tr_mode, reduce_weight_num=len(threshold))
            self.explainer.fit(input_ant, input_con, output)
        fs = self.explainer.feature_attribution(input_ant)  # [:,0,:].unsqueeze(dim=1)
        self.score = self.explainer.score(input_ant, input_con, output)
        fs = np.array(fs.detach().numpy())

        # generate explanations
        fs = positive_normalize(fs)  # normalize the salience value for each superpixel
        if tr_mode == 1:
            fs_semantic = np.dot(membership_semantic[0], fs * superpixel_size) / np.transpose(semantic_sum)  # calculate the salience value for each semantic segment
            fs_semantic = np.where(np.isnan(fs_semantic), 0, fs_semantic)
        else:
            fs_semantic = []  # calculate the salience value for each semantic segment
            for j, _ in enumerate(threshold):
                fs_semantic.append(
                    np.dot(membership_semantic[j], fs * superpixel_size) / np.transpose(semantic_sum))
            fs_semantic = np.where(np.isnan(fs_semantic), 0, fs_semantic)
            fs_semantic = np.average(fs_semantic, axis=0,
                                     weights=np.maximum(np.array(self.explainer.type_reduce.data), 0))
        self.fs = fs
        self.fs_semantic = fs_semantic
        self.max_label = np.argmax(fs_semantic) + 1  # the semantic segment contribute to the classification result

        # deliver explanations
        if require_rules or require_semantic:  # calculate the classification scores for IF-THEN rules
            model_output, fir_str_bar, _ = self.explainer.forward(input_ant, input_con)
            fir_str_bar_bar = fir_str_bar / torch.sum(fir_str_bar, 0).unsqueeze(0)
            self.c_score = torch.einsum('MC,MR->CR', model_output, fir_str_bar_bar).data.numpy()
            if class_list is None:
                self.c_score = np.squeeze(self.c_score, axis=0)
            if require_semantic:
                return input_con.numpy(), model_output
        if require_dataset:
            return input_con, output
        if class_list is not None:
            return None
        mask = np.array(np.array(semantic_map) == self.max_label)
        if improve is True:
            mask = remove_small_regions(mask)  # remove the isolated small regions in explanation
            mask = binary_fill_holes(mask)  # fill the holes in explanation
        return mask  # a binary metrix with the same size of the image instance, where explanation is True while other part is False
