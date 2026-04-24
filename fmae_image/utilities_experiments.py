import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import copy
from skimage.io import imread
from skimage import transform
from skimage import measure
from scipy import ndimage
import cv2
from skimage.segmentation import slic, mark_boundaries
import random
import glob

def input_transform():
    # ImageNet-style transformation
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def compress_image(img):
    # compress the image to a manageable size
    height, width = img.shape[:2]
    if width < height: # < by short side or > by long side
        new_width = 256
        new_height = int((256 / width) * height)
    else:
        new_height = 256
        new_width = int((256 / height) * width)
    resized_img = transform.resize(img, (new_height, new_width), anti_aliasing=True)
    return (resized_img * 255).astype('uint8')


def resize_like(paths, img_a):  # for Anchor
    out = []
    for i, path in enumerate(paths):
        image_shape = img_a.shape
        img = imread(path)
        img_resized = transform.resize(img, image_shape, anti_aliasing=True)
        out.append(img_resized)
    return out


def seg_idx_to_mask(img, seg, idx):  # for Anchor
    mask_ = img[:, :, 0].copy()
    mask_[:] = 0
    for x in idx:
        mask_[seg == x] = 1
    return mask_


def ab_mask(mask_, img_a, img_b=255):
    # combine 2 images by mask
    out_img = copy.deepcopy(img_a)
    out_img[mask_ == 0] = img_b
    return out_img


def add_jag_iteratively(mask, max_iter=10, prob=0.5):
    # perturb the boundary of a mask
    perturbed = mask.copy()
    for _ in range(max_iter):
        # expand
        dilated = ndimage.binary_dilation(perturbed)

        # new boundary
        new_layer = np.logical_and(dilated, ~perturbed)

        # sample
        select = np.random.rand(*new_layer.shape) < prob
        add_pixels = np.logical_and(new_layer, select)

        # add
        perturbed[add_pixels] = True

    return perturbed


def add_jag_per_contour(mask, step=10, min_len=4, max_len=8):
    # perturb the boundary of a mask
    contours = measure.find_contours(mask.astype(np.uint8), 0.5)
    if len(contours) == 0:
        return mask.copy()

    h, w = mask.shape
    new_mask = np.zeros_like(mask, dtype=np.uint8)

    for contour in contours:
        contour = contour[:, ::-1]  # (row,col) -> (x,y)
        contour = contour[::step]  # sample
        new_points = []

        for i in range(len(contour)):
            p = contour[i]
            prev_p = contour[i - 1]
            next_p = contour[(i + 1) % len(contour)]
            tangent = next_p - prev_p
            normal = np.array([-tangent[1], tangent[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-6)

            length = np.random.randint(min_len, max_len)
            new_p = p + normal * length

            # check inside or outside
            x, y = int(np.clip(new_p[0], 0, w - 1)), int(np.clip(new_p[1], 0, h - 1))
            if mask[y, x]:
                new_p = p - normal * length

            new_points.append(p)
            new_points.append(new_p)

        new_points = np.array(new_points).astype(np.int32)
        cv2.fillPoly(new_mask, [new_points], 1)

    return new_mask.astype(bool)


def generate_random_superpixel_images(target_img, folder_path, n_random_images=100, n_segments=50, visualize=False):
    # generate a set of random images by the superpixel segmentation of a given image
    h, w, _ = target_img.shape
    segments = slic(target_img, n_segments=n_segments, compactness=10, start_label=0)
    # segments = quickshift(target_img, kernel_size=3, max_dist=6, ratio=0.5)
    unique_segments = np.unique(segments)
    s = len(unique_segments)

    # load and resize
    all_imgs_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    preloaded_imgs = []
    for path in all_imgs_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
        preloaded_imgs.append(img)

    # generate random images by mask
    masks = [(segments == sp_id) for sp_id in unique_segments]

    generated_images = []
    for _ in range(n_random_images):
        random_imgs = random.sample(preloaded_imgs, s)
        new_img = np.zeros_like(target_img)
        for i, mask in enumerate(masks):
            new_img[mask] = random_imgs[i][mask]
        generated_images.append(new_img)

    # visualize result
    if visualize and len(generated_images) > 0:
        plt.figure(figsize=(10, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(target_img)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        mb = mark_boundaries(target_img, segments)
        plt.imsave(f"result/samples/superpixel.png", mb)
        plt.imshow(mb)
        plt.title(f"Superpixels (s={s})")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(generated_images[0])
        plt.title("Example Image")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return generated_images


def save_images(images, save_dir='result/samples', prefix='img'):
    # save a set of images
    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(images):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        filename = os.path.join(save_dir, f'{prefix}_{i:03d}.png')
        cv2.imwrite(filename, img)
    print(f"Saved {len(images)} images to '{save_dir}'.")


def load_images_from_dir(load_dir='output_images', prefix='img'):
    # load a set of images with given prefix
    image_paths = sorted(glob.glob(os.path.join(load_dir, f'{prefix}_*.png')))
    images = []
    for path in image_paths:
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)
    print(f"Loaded {len(images)} images from '{load_dir}'.")
    return images

