import cv2
import os
import matplotlib.pyplot as plt
import tempfile
TEMP_SAVE_PATH = os.path.join(tempfile.gettempdir(), "denoised_temp.jpg")

def enhanced_distortion_filters(img, distortion_type=None):
    img = cv2.resize(img, (160, 160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if distortion_type is None:
        return img

    distortion_type = distortion_type.lower()
    if "foggy" in distortion_type:
        img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    elif "sunlight" in distortion_type or "lowlight" in distortion_type:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5)
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    elif "blur" in distortion_type:
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.detailEnhance(img, sigma_s=5, sigma_r=0.15)
    return img

def denoise_if_needed(image_path, distortion_type):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if distortion_type in ['noise', 'rain']:
        denoised = cv2.medianBlur(img_rgb, 5)
        denoised_bgr = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
        cv2.imwrite(TEMP_SAVE_PATH, denoised_bgr)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title('Original (Distorted)')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(denoised)
        plt.title('Denoised Output')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return TEMP_SAVE_PATH
    else:
        return image_path