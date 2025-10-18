import cv2
import numpy as np
import os
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.embedding import get_arcface_embedding


# ---------- CONFIG ----------
THRESHOLD = 0.5  # similarity threshold for declaring a match
FONT = cv2.FONT_HERSHEY_SIMPLEX

print("Enter the full path to your reference image:")
ref_path = input("Path: ").strip()


if not os.path.exists(ref_path):
    print("Reference image not found. Exiting.")
    exit()


ref_img = cv2.imread(ref_path)
if ref_img is None:
    print("Could not load image.")
    exit()


print("Generating embedding for reference image...")
ref_embedding = get_arcface_embedding(ref_img)
if ref_embedding is None:
    print("No face detected in reference image.")
    exit()
print("Reference face embedding saved.\n")


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera.")
    exit()


print("Press 'c' to capture a live image and compare.")
print("Press 'q' to quit without comparing.\n")


captured_embedding = None


while True:
    ret, frame = cap.read()
    if not ret:
        break


    cv2.imshow("Live Camera - Press 'c' to Capture", frame)
    key = cv2.waitKey(1) & 0xFF


    if key == ord('c'):
        captured_embedding = get_arcface_embedding(frame)
        if captured_embedding is None:
            print("No face detected in captured frame.")
            continue


        # Comparing
        similarity = np.dot(ref_embedding, captured_embedding)
        print(f"\n Similarity Score: {similarity:.3f}")


        if similarity >= THRESHOLD:
            print("Faces Match (Same Identity)")
            cv2.putText(frame, f" Match! ({similarity:.3f})", (50,50),
                        FONT, 1, (0,255,0), 2)
        else:
            print("Faces Do Not Match (Different Person)")
            cv2.putText(frame, f"Mismatch ({similarity:.3f})", (50,50),
                        FONT, 1, (0,0,255), 2)


        cv2.imshow("Comparison Result", frame)


    elif key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
print("\nProgram terminated.")



