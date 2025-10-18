import cv2
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