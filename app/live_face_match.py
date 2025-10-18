import cv2
import numpy as np
import os
from utils.embedding import get_arcface_embedding




# -------- CONFIG --------
BASE_THRESHOLD = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX




# ---------- QUALITY METRICS ----------
def estimate_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)




def estimate_noise(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.std()




# ---------- SMART ENHANCEMENT ----------
def enhance_image(img, brightness=None, noise=None):
    enhanced = img.copy().astype(np.uint8)


    # CLAHE for brightness / contrast correction
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clip_limit = 3.0 if brightness and brightness < 70 else 2.0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


    # Noise handling (denoise only if actual noise present)
    if noise and noise > 30:
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)


    # Mild detail enhancement, not blur
    enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)


    return enhanced




# ---------- FILTER SIMULATION ----------
def apply_custom_filter(img, filter_type="none"):
    if filter_type == "none":
        return img
    elif filter_type == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)
    elif filter_type == "sepia":
        sepia = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        return cv2.transform(img, sepia)
    elif filter_type == "contrast":
        return cv2.convertScaleAbs(img, alpha=1.3, beta=20)
    elif filter_type == "blur":
        return cv2.GaussianBlur(img, (5, 5), 0)
    elif filter_type == "noise":
        noisy = img.copy().astype(np.float32)
        h, w, c = noisy.shape

        # 1️⃣ Gaussian noise (random variations)
        noise_strength = 40  # higher value = stronger noise
        gaussian_noise = np.random.normal(0, noise_strength, (h, w, c))
        noisy += gaussian_noise

        # 2️⃣ Salt-and-pepper noise (random black & white pixels)
        sp_ratio = 0.05  # 5% of pixels affected; increase for denser noise
        sp_mask = np.random.rand(h, w) < sp_ratio
        noisy[sp_mask] = np.random.choice([0, 255], size=(sp_mask.sum(), c))

        # Clip values to valid range
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        return noisy

    elif filter_type == "fog":
        fog = np.full_like(img, 255)
        return cv2.addWeighted(img, 0.7, fog, 0.3, 0)
    elif filter_type == "rain":
        
        rain = img.copy()
        overlay = rain.copy()
        h, w = rain.shape[:2]

        # Increase number of drops for denser rain
        num_drops = 1000  # heavier rain

        for _ in range(num_drops):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            length = np.random.randint(10, 20)  # longer drops
            thickness = np.random.randint(2, 4)  # thicker drops
            brightness = np.random.randint(200, 255)  # brighter drops
            x_end = x + np.random.randint(-1, 2)
            y_end = y + length
            cv2.line(overlay, (x, y), (x_end, y_end), (brightness, brightness, brightness), thickness)

        # Blend overlay for smoother effect
        rain = cv2.addWeighted(rain, 0.7, overlay, 0.3, 0)

        # Vertical motion blur for realism
        ksize = 3
        kernel = np.zeros((ksize, ksize))
        kernel[:, ksize // 2] = np.ones(ksize) / ksize
        rain = cv2.filter2D(rain, -1, kernel)

        return rain

    elif filter_type == "lowlight":
        return cv2.convertScaleAbs(img, alpha=0.6, beta=-25)
    else:
        return img




# ---------- TEST-TIME AUGMENTATION ----------
def get_tta_embeddings(img, adaptive=False):
    variations = [img]
    if adaptive:
        brighter = cv2.convertScaleAbs(img, alpha=1.3, beta=15)
        darker = cv2.convertScaleAbs(img, alpha=0.8, beta=-15)
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 15, 15, 7, 21)
        variations += [brighter, darker, blurred, denoised]
    embeddings = []
    for var in variations:
        emb = get_arcface_embedding(var)
        if emb is not None:
            embeddings.append(emb)
    return np.mean(embeddings, axis=0) if embeddings else None




# ---------- SIMILARITY ----------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))




def adaptive_threshold(brightness, noise):
    threshold = BASE_THRESHOLD
    if brightness < 70:
        threshold -= (70 - brightness) / 200
    if noise > 30:
        threshold -= (noise - 30) / 200
    return max(0.4, min(0.6, threshold))




def is_match(emb1, emb2, threshold):
    sim = cosine_similarity(emb1, emb2)
    print(f"→ Similarity: {sim:.3f} | Threshold: {threshold:.3f}")
    return sim >= threshold




# ---------- DISPLAY ----------
def side_by_side_display(img1, img2, target_height=400, gap=40):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    scale1 = target_height / h1
    scale2 = target_height / h2
    img1_resized = cv2.resize(img1, (int(w1 * scale1), target_height))
    img2_resized = cv2.resize(img2, (int(w2 * scale2), target_height))
    gap_img = np.ones((target_height, gap, 3), dtype=np.uint8) * 255
    return cv2.hconcat([img1_resized, gap_img, img2_resized])




# ---------- MAIN ----------
ref_path = input("Enter full path to reference image: ").strip()
if not os.path.exists(ref_path):
    print("❌ Reference image not found.")
    exit()


ref_img = cv2.imread(ref_path)
if ref_img is None:
    print("❌ Could not load reference image.")
    exit()


# Keep original reference for display
original_ref_img = ref_img.copy()


# Auto-enhance reference image for embedding only
ref_brightness = estimate_brightness(ref_img)
ref_noise = estimate_noise(ref_img)
enhanced_ref_img = enhance_image(ref_img.copy(), ref_brightness, ref_noise)
ref_emb = get_tta_embeddings(enhanced_ref_img, adaptive=True)


print("✅ Reference embedding captured. Starting live camera...")
print("Press 'c' to capture | 'q' to quit.")
print("Filters: 1-None 2-Sharpen 3-Sepia 4-Contrast 5-Blur 6-Noise 7-Rain 8-Fog 9-Lowlight\n")


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera.")
    exit()


filter_options = {
    ord('1'): "none",
    ord('2'): "sharpen",
    ord('3'): "sepia",
    ord('4'): "contrast",
    ord('5'): "blur",
    ord('6'): "noise",
    ord('7'): "rain",
    ord('8'): "fog",
    ord('9'): "lowlight"
}
selected_filter = "none"


while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Could not read frame.")
        break


    # Keep original live frame for display
    original_live = frame.copy()


    filtered_frame = apply_custom_filter(frame.copy(), selected_filter)
    display = filtered_frame.copy()
    cv2.putText(display, f"Filter: {selected_filter} | 'c' Capture | 'q' Quit",
                (10, 30), FONT, 0.6, (0, 255, 255), 2)
    cv2.imshow("Live Camera", display)


    key = cv2.waitKey(1) & 0xFF


    if key in filter_options:
        selected_filter = filter_options[key]
    elif key == ord('q'):
        print("👋 Exiting.")
        break
    elif key == ord('c'):
        print("⚙️ Processing capture (Enhancement + Adaptive Threshold)...")


        # Step 1: Enhance captured frame for embedding only (no filter)
        live_brightness = estimate_brightness(original_live)
        live_noise = estimate_noise(original_live)
        enhanced_live = enhance_image(original_live.copy(), live_brightness, live_noise)


        # Step 2: Adaptive threshold
        threshold = adaptive_threshold(live_brightness, live_noise)


        # Step 3: Get embeddings
        live_emb = get_tta_embeddings(enhanced_live, adaptive=True)
        if live_emb is None:
            print("❌ Could not extract embedding.")
            continue


        # Step 4: Match
        match = is_match(ref_emb, live_emb, threshold)
        label = "MATCH FOUND " if match else "NOT A MATCH "
        color = (0, 255, 0) if match else (0, 0, 255)


        # Step 5: Apply filter for display only
        filtered_display = apply_custom_filter(original_live.copy(), selected_filter)
        cv2.putText(filtered_display, label, (50, 50), FONT, 1.0, color, 3)


        # Step 6: Display side-by-side: original reference vs filtered live
        side_by_side = side_by_side_display(original_ref_img, filtered_display)
        cv2.imshow("Reference vs Live", side_by_side)
        print("⏱ Displaying result for 5 seconds...")
        cv2.waitKey(5000)
        cv2.destroyWindow("Reference vs Live")




cap.release()
cv2.destroyAllWindows()



