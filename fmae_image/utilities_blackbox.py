import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes
import sklearn
from functools import partial


def get_predicted_label(labels, class_num=1):
    # find the top class_num predicted class
    np_matrix = np.array(labels)
    top = np.argsort(labels[0])[-class_num:]
    top_labels = list(top)
    top_labels.reverse()
    np_matrix = np_matrix[:, top_labels]
    return np_matrix, top_labels


def draw_overlay_heatmap(f_ori, superpixel, fs, Label=None, max_label=None, vmin=None, vmax=None, save_path=None, simplify=False, bar=True):
    # generate and save the salience map
    if fs.ndim == 1:
        fs = np.expand_dims(fs, axis=1)
    heatmap = fs[:, 0][np.array(superpixel).astype(np.int16)-1]
    f_edge = f_ori.copy()
    if Label is not None:
        idx1 = np.where(Label != max_label, 0, 1)
        if simplify is True:
            idx1 = remove_small_regions(idx1)
            idx1 = binary_fill_holes(idx1)
        edges = cv2.Canny(idx1.astype(np.uint8) * 255, 100, 200)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges.astype(np.uint8), kernel, iterations=1)
        f_edge[edges != 0] = [255, 0, 0]

    fig, ax = plt.subplots()
    # ax.imshow(heatmap, alpha=alpha)
    ax.imshow(f_edge)
    ax.set_xticks([])
    ax.set_yticks([])
    map = ax.imshow(heatmap, vmin=vmin, vmax=vmax, alpha=0.5)
    if bar is True:
        cbar = plt.colorbar(map, ax=ax, shrink=1.0, aspect=20)
        cbar.ax.set_position([cbar.ax.get_position().x0,
                              ax.get_position().y0,
                              cbar.ax.get_position().width,
                              ax.get_position().height])
        cbar.ax.tick_params(labelsize=0)
        cbar.set_ticks([])
        # cbar.set_label('Salience')

    if save_path is None:
        plt.show(block=True)
    else:
        # plt.show(block=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)  # save_name=task_name+'_'+name
    plt.close(fig)
    return heatmap


def positive_normalize(value, axis=0, mode='norm', pos=True):
    # keeping the positive values and normalization
    if pos:
        value = np.where(value < 0, 0, value)
    if mode == '0-1':
        # normalize_value = (value - np.min(value, axis=axis)) / (np.max(value, axis=axis) - np.min(value, axis=axis))
        min_vals = value.min(axis=axis, keepdims=True)
        max_vals = value.max(axis=axis, keepdims=True)
        normalize_value = (value - min_vals) / (max_vals - min_vals)
    elif mode == 'std':
        mean = value.mean(axis=axis, keepdims=True)
        std = value.std(axis=axis, keepdims=True)
        normalize_value = (value - mean) / std
    else:
        normalize_value = value / np.linalg.norm(value, axis=axis, keepdims=True)
    return normalize_value


def print_rules(rule_base, num_fuzzy_set, c_score, save_path='rules.txt',
                feature_list=None, class_list=None, fuzzy_set_list=None,
                std=None, is_print=False):
    # format and save the IF-THEN rules
    rule_base = np.array(rule_base)
    if fuzzy_set_list is None:
        if num_fuzzy_set == 3:
            fuzzy_set_list = ['low', 'medium', 'high']
        elif num_fuzzy_set == 2:
            fuzzy_set_list = ['inactive', 'active']
        else:
            fuzzy_set_list = ['Level {}'.format(i) for i in range(1, num_fuzzy_set + 1)]
    if class_list is None:
        feature_list = ['S{}'.format(i) for i in range(1, rule_base.shape[1] + 1)]
    if class_list is None:
        class_list = ['C{}'.format(i) for i in range(1, c_score.shape[0] + 1)]

    with open(save_path, 'w') as f:
        for i in range(rule_base.shape[0]):
            rule = 'r'+str(i+1)+': IF '
            for id, word in enumerate(feature_list):
                if rule_base[i, id] != num_fuzzy_set:
                    rule = rule + '{} is {}, '.format(word, fuzzy_set_list[rule_base[i, id]])
            rule = rule + 'THEN predicted as '
            for id, word in enumerate(class_list):
                rule = rule + '{} with {:.3f}'.format(word, c_score[id, i])
                if std is not None:
                    rule = rule + '\u00B1{:.3f}'.format(std[id, i])
                if id == len(class_list) - 1:
                    rule = rule + ';'
                else:
                    rule = rule + ', '
            f.write(rule)
            f.write('\n')

            if is_print:
                print(rule)


def remove_small_regions(mask):
    # remove the redundant regions from explanation
    mask = mask.astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels == 2:
        return mask.astype(bool)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    new_mask = np.zeros_like(mask)
    new_mask[labels == largest_label] = 255
    return new_mask.astype(bool)


def generate_binary_matrix(index_lists, max_index):
    # Generate a binary matrix based on the given index lists.
    # Number of index lists (rows)
    m = len(index_lists)
    # Initialize a binary matrix with zeros
    binary_matrix = np.zeros((m, max_index), dtype=int)
    # Fill the binary matrix based on index_lists
    for i, indices in enumerate(index_lists):
        binary_matrix[i, indices] = 1
    return binary_matrix


def binary_matrix_to_decimal_vector(matrix):
    return [int("".join(map(str, row)), 2) for row in matrix]


def single_bar(data, scatter=True, save_path=None):
    # bar graph for domain explanation
    mean = np.mean(data, axis=0)  # 按列计算均值
    std = np.std(data, axis=0)
    num_columns = data.shape[1]

    x_pos = np.arange(num_columns)
    plt.figure(figsize=(3, 4))
    plt.bar(x_pos, mean, capsize=5, color='skyblue', alpha=0.7)  # yerr=std,

    if scatter:
        num_rows = data.shape[0]
        colors = plt.cm.viridis(np.linspace(0, 1, num_rows))
        for ii in range(num_rows):
            jitter = np.random.normal(0, 0.02, size=num_columns)
            plt.scatter(x_pos+jitter, data[ii, :], color=colors[ii], alpha=0.6,
                        )  # label=None

    ax = plt.gca()
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'S {ii + 1}' for ii in range(num_columns)], fontsize=14)
    ax.set_ylabel('Normalized Salience Value', fontsize=16)
    ax.set_xlabel('Semantic Features', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path+'bar_plot.png', bbox_inches='tight', dpi=300)
    plt.close()


def get_weights(data, kernel_width=0.5, flag=True):
    # weight the samples by distance (from LIME)
    def kernel(d, kernel_width):
        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

    if flag is True:
        kernel_fn = partial(kernel, kernel_width=kernel_width)
        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric='euclidean'
        ).ravel()
        return kernel_fn(distances)
    else:
        return None
