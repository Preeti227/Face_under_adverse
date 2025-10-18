import glob
from itertools import combinations
import os
import random

import cv2
from matplotlib import pyplot as plt

def generate_pairs_from_reference(ref_dir, max_negative_pairs_per_identity=5):
    identity_folders = sorted([f for f in os.listdir(ref_dir) if os.path.isdir(os.path.join(ref_dir, f))])
    id_to_images = {
        identity: glob(os.path.join(ref_dir, identity, "*.jpg"))
        for identity in identity_folders
    }

    pairs = []

    for identity, images in id_to_images.items():
        if len(images) < 2:
            continue
        for img1, img2 in combinations(images, 2):
            pairs.append((img1, img2, 1))

    for identity, img_list in id_to_images.items():
        if not img_list:
            continue
        other_identities = [id2 for id2 in identity_folders if id2 != identity and id_to_images[id2]]
        if not other_identities:
            continue
        for _ in range(min(max_negative_pairs_per_identity, len(img_list))):
            img1 = random.choice(img_list)
            neg_id = random.choice(other_identities)
            img2 = random.choice(id_to_images[neg_id])
            pairs.append((img1, img2, 0))

    pos_count = sum(1 for _, _, l in pairs if l == 1)
    neg_count = sum(1 for _, _, l in pairs if l == 0)
    print(f"Total pairs: {len(pairs)} | Positive: {pos_count} | Negative: {neg_count}")

    return pairs

def show_image_pair(img1_path, img2_path, title1, title2, label_text):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print(f"Failed to load: {img1_path} or {img2_path}")
        return
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title(title2)
    plt.axis('off')

    plt.suptitle(label_text, fontsize=16)
    plt.tight_layout()
    plt.show()
    
def show_positive_pairs(pairs, num_to_show=5):
    pos_pairs = [pair for pair in pairs if pair[2] == 1]
    for i, (img1, img2, _) in enumerate(pos_pairs[:num_to_show]):
        show_image_pair(img1, img2, "Image 1", "Image 2", f"Positive Pair #{i+1} (Label=1)")

def show_negative_pairs(pairs, num_to_show=5): 
    neg_pairs = [pair for pair in pairs if pair[2] == 0]
    for i, (img1, img2, _) in enumerate(neg_pairs[:num_to_show]):
        show_image_pair(img1, img2, "Image 1", "Image 2", f"Negative Pair #{i+1} (Label=0)")