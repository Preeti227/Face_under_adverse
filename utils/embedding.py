import glob
import os
import cv2
import numpy as np
from .preprocessing import enhanced_distortion_filters
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(160, 160))

def get_arcface_embedding(img, distortion_type=None):
    img = cv2.resize(img, (160, 160))
    if distortion_type and any(x in distortion_type for x in ["noise", "rain"]):
        img = cv2.medianBlur(img, 5)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5)
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    else:
        img = enhanced_distortion_filters(img, distortion_type)

    img_flip = cv2.flip(img, 1)
    faces1 = app.get(img)
    faces2 = app.get(img_flip)

    emb_list = []
    if faces1:
        emb1 = faces1[0].embedding
        emb_list.append(emb1 / np.linalg.norm(emb1))
    if faces2:
        emb2 = faces2[0].embedding
        emb_list.append(emb2 / np.linalg.norm(emb2))
    if not emb_list:
        return None
    return np.mean(emb_list, axis=0)

def cache_reference_embeddings(reference_dir):
    ref_embeddings = {}
    for identity in sorted(os.listdir(reference_dir)):
        ref_folder = os.path.join(reference_dir, identity)
        ref_images = glob(os.path.join(ref_folder, "*.jpg"))
        emb_list = []
        for ref_path in ref_images:
            ref_img = cv2.imread(ref_path)
            if ref_img is None:
                continue
            ref_faces = app.get(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
            if not ref_faces:
                continue
            emb = ref_faces[0].embedding
            emb = emb / np.linalg.norm(emb)
            emb_list.append(emb)
        if emb_list:
            ref_embeddings[identity] = emb_list
    return ref_embeddings
