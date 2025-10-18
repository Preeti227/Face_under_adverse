import cv2
import numpy as np
import os
from utils.embedding import get_arcface_embedding


BASE_THRESHOLD = 0.5  # base similarity threshold
FONT = cv2.FONT_HERSHEY_SIMPLEX


def estimate_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def estimate_noise(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.std()  # higher = noisier


def enhance_image(img, brightness=None, noise=None):
    enhanced = img.copy().astype(np.uint8)


    # Convert to LAB color space for better brightness/contrast adjustment
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)


    # CLAHE on L channel (brightness) only
    clip_limit = 3.0 if brightness and brightness < 70 else 2.0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)


    # Merge back and convert to BGR
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


    # Denoising intensity based on noise level
    h = 15 if noise and noise > 35 else 10
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, h, h, 7, 21)


    # Slight Gaussian blur for smoothing
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)


    return enhanced

# ---------- CUSTOM FILTERS ----------
def apply_custom_filter(img, filter_type="none"):
    if filter_type == "sharpen":
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)
    elif filter_type == "sepia":
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        return cv2.transform(img, sepia_filter)
    elif filter_type == "contrast":
        return cv2.convertScaleAbs(img, alpha=1.3, beta=20)
    elif filter_type == "blur":
        return cv2.GaussianBlur(img, (5,5), 0)
    else:
        return img


# ---------- TEST-TIME AUGMENTATION ----------
def get_tta_embeddings(img, adaptive=False):
    """
    Generate embeddings with optional adaptive augmentations.
    All augmentations preserve color.
    """
    variations = [img]


    if adaptive:
        # Brighter
        brighter = cv2.convertScaleAbs(img, alpha=1.3, beta=15)
        # Darker
        darker = cv2.convertScaleAbs(img, alpha=0.8, beta=-15)
        # Slight blur
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        # Denoised
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 15, 15, 7, 21)


        variations += [brighter, darker, blurred, denoised]


    embeddings = []
    for var in variations:
        emb = get_arcface_embedding(var)
        if emb is not None:
            embeddings.append(emb)


    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return None




# ---------- SIMILARITY CHECK ----------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def adaptive_threshold(brightness, noise):
    """Adjust threshold dynamically based on image quality."""
    threshold = BASE_THRESHOLD
    # Darker images = lower threshold
    if brightness < 70:
        threshold -= (70 - brightness) / 200  # more responsive
    # Noisy images = lower threshold
    if noise > 30:
        threshold -= (noise - 30) / 200
    return max(0.4, min(0.6, threshold))


def is_match(emb1, emb2, threshold):
    sim = cosine_similarity(emb1, emb2)
    print(f"→ Similarity: {sim:.3f} | Threshold: {threshold:.3f}")
    return sim >= threshold


# ---------- MAIN LOGIC ----------
ref_path = input("Enter full path to reference image: ").strip()
if not os.path.exists(ref_path):
    print("❌ Reference image not found. Exiting.")
    exit()


ref_img = cv2.imread(ref_path)
if ref_img is None:
    print("❌ Could not load reference image.")
    exit()


ref_brightness = estimate_brightness(ref_img)
ref_noise = estimate_noise(ref_img)
ref_img = enhance_image(ref_img, ref_brightness, ref_noise)
ref_emb = get_tta_embeddings(ref_img, adaptive=True)


print("✅ Reference embedding captured. Starting live camera...")
print("Press 'c' to capture a live image and compare.")
print("Press 'q' to quit without comparing.\n")


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera.")
    exit()
   
# ---------- FILTER OPTIONS ----------
filter_options = {
    ord('1'): "none",
    ord('2'): "sharpen",
    ord('3'): "sepia",
    ord('4'): "contrast",
    ord('5'): "blur"
}
selected_filter = "none"  # default
# ---------- FUNCTION TO CONCATENATE IMAGES SAFELY ----------
def side_by_side_display(img1, img2, target_height=400):
    # Resize both images to the same height
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]


    scale1 = target_height / h1
    scale2 = target_height / h2


    new_w1 = int(w1 * scale1)
    new_w2 = int(w2 * scale2)


    img1_resized = cv2.resize(img1, (new_w1, target_height))
    img2_resized = cv2.resize(img2, (new_w2, target_height))


    # Ensure both are 3-channel BGR
    if len(img1_resized.shape) == 2:
        img1_resized = cv2.cvtColor(img1_resized, cv2.COLOR_GRAY2BGR)
    if len(img2_resized.shape) == 2:
        img2_resized = cv2.cvtColor(img2_resized, cv2.COLOR_GRAY2BGR)


    return cv2.hconcat([img1_resized, img2_resized])




while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Could not read frame.")
        break


    display = frame.copy()
    cv2.putText(display, f"Press 'c' to Capture | 'q' to Quit | Filter: {selected_filter}",
                (10, 30), FONT, 0.6, (0, 255, 255), 2)
    cv2.putText(display, "1:None 2:Sharpen 3:Sepia 4:Contrast 5:Blur",
                (10, 60), FONT, 0.5, (255, 255, 0), 1)
    cv2.imshow("Live Camera", display)


    key = cv2.waitKey(1) & 0xFF


    # Switch filters in real-time
    if key in filter_options:
        selected_filter = filter_options[key]


    if key == ord('q'):
        print("👋 Exiting.")
        break
    elif key == ord('c'):
        live_brightness = estimate_brightness(frame)
        live_noise = estimate_noise(frame)
        threshold = adaptive_threshold(live_brightness, live_noise)


        live_img = enhance_image(frame, live_brightness, live_noise)
        live_img = apply_custom_filter(live_img, selected_filter)
        live_emb = get_tta_embeddings(live_img, adaptive=True)


        if live_emb is None:
            print("Could not extract embedding.")
            continue


        match = is_match(ref_emb, live_emb, threshold)
        label = "MATCH FOUND " if match else "NOT A MATCH "
        color = (0, 255, 0) if match else (0, 0, 255)
        cv2.putText(live_img, label, (50, 50), FONT, 1.0, color, 3)
       
            # Show match result in console
        if match:
            print("MATCH FOUND! Same Person.")
        else:
            print("NOT A MATCH.")


        # Side-by-side display until key pressed
        side_by_side = side_by_side_display(ref_img, live_img)
        cv2.imshow("Reference vs Live", side_by_side)
        print("Press any key on the image window to continue...")
        cv2.waitKey(0)  # waits indefinitely until a key is pressed
        cv2.destroyWindow("Reference vs Live")  # closes side-by-side window



