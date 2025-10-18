import glob
from itertools import combinations
import os
import random

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