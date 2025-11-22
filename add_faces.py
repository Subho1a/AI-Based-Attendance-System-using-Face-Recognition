
import cv2
import os
import pickle
import numpy as np
import time

# -------------------- Debug info --------------------
print("Working directory:", os.getcwd())

# -------------------- Ensure folders & pkl exist --------------------
if not os.path.exists("data"):
    os.makedirs("data")

if not os.path.exists("data/names.pkl"):
    with open("data/names.pkl", "wb") as f:
        pickle.dump([], f)

if not os.path.exists("data/faces_data.pkl"):
    with open("data/faces_data.pkl", "wb") as f:
        pickle.dump(np.array([]), f)

# -------------------- Load current data --------------------
with open("data/names.pkl", "rb") as f:
    names = pickle.load(f)

with open("data/faces_data.pkl", "rb") as f:
    faces_data = pickle.load(f)

# -------------------- Get the user name --------------------
name = input("Enter your name (no spaces recommended): ").strip()
if name == "":
    print("Name cannot be empty. Exiting.")
    exit()

# Create samples folder for debugging images
samples_dir = os.path.join("data", "samples", name)
os.makedirs(samples_dir, exist_ok=True)
print("Sample images will be saved to:", samples_dir)

# -------------------- Initialize camera & cascade --------------------
# Use CAP_DSHOW on Windows for more reliable camera access
try:
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
except TypeError:
    # fallback if cv2 doesn't support CAP_DSHOW in this build
    video = cv2.VideoCapture(0)

time.sleep(0.5)
if not video.isOpened():
    print("ERROR: Could not open camera. Try changing the camera index (0 -> 1).")
    exit()

haarcascade_path = os.path.join("data", "haarcascade_frontalface_default.xml")
if not os.path.exists(haarcascade_path):
    print(f"ERROR: Haarcascade file not found at {haarcascade_path}")
    print("Please download 'haarcascade_frontalface_default.xml' and place it inside the data/ folder.")
    video.release()
    exit()

facedetect = cv2.CascadeClassifier(haarcascade_path)
if facedetect.empty():
    print("ERROR: Failed to load cascade. Check the XML file.")
    video.release()
    exit()

print("Camera opened and cascade loaded successfully.")

# -------------------- Capture params --------------------
TARGET_COUNT = 100           # number of face images to capture
MIN_FACE_SIZE = (50, 50)     # min face size to accept
counter = 0
new_face_data = []

# To reduce duplicates, we can skip frames so we don't capture successive identical frames:
SKIP_FRAMES = 3
skip = 0

print("\n📸 Capturing face images... Look at the camera.")
print("➡ Press Q anytime to exit.\n")

while True:
    ret, frame = video.read()
    if not ret or frame is None:
        print("WARNING: Empty frame received. Skipping...")
        time.sleep(0.1)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # optional: histogram equalization can sometimes help detection
    gray_eq = cv2.equalizeHist(gray)

    # detectMultiScale tuned for better detection on varied lighting
    faces = facedetect.detectMultiScale(
        gray_eq,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=MIN_FACE_SIZE
    )

    # Draw UI overlay
    cv2.putText(frame, f"Captured: {counter}/{TARGET_COUNT}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, "Press Q to exit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if len(faces) == 0:
        cv2.imshow("Capturing Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture stopped manually by user.")
            break
        continue

    # If multiple faces, pick the largest (most likely subject)
    faces = sorted(faces, key=lambda rect: rect[2] * rect[3], reverse=True)
    (x, y, w, h) = faces[0]

    # draw rectangle
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # skip some frames to reduce near-duplicates
    if skip < SKIP_FRAMES:
        skip += 1
        cv2.imshow("Capturing Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture stopped manually by user.")
            break
        continue
    skip = 0

    # Crop & resize to fixed size and flatten
    crop = frame[y:y+h, x:x+w]
    try:
        resized = cv2.resize(crop, (50, 50))
    except Exception as e:
        print("Resize failed:", e)
        continue

    new_face_data.append(resized.flatten())
    counter += 1

    # Save sample image for debugging
    sample_path = os.path.join(samples_dir, f"{int(time.time()*1000)}_{counter}.jpg")
    cv2.imwrite(sample_path, resized)

    # Show capture feedback
    cv2.putText(frame, f"Captured: {counter}/{TARGET_COUNT}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Capturing Faces", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Capture stopped manually by user.")
        break

    if counter >= TARGET_COUNT:
        print(f"✔ Captured {TARGET_COUNT} images for '{name}'")
        break

# cleanup
video.release()
cv2.destroyAllWindows()

# -------------------- Save data --------------------
if len(new_face_data) == 0:
    print("No new face data captured. Exiting without saving.")
    exit()

new_face_data = np.array(new_face_data)
print("New face data shape:", new_face_data.shape)

# If faces_data is empty array (size 0) we replace it
if isinstance(faces_data, np.ndarray) and faces_data.size != 0:
    try:
        faces_data = np.vstack([faces_data, new_face_data])
    except Exception as e:
        print("Error stacking arrays:", e)
        # fallback: just overwrite
        faces_data = new_face_data
else:
    faces_data = new_face_data

# Append names (one name per captured sample)
names.extend([name] * new_face_data.shape[0])

# Save back to pkl
with open("data/faces_data.pkl", "wb") as f:
    pickle.dump(faces_data, f)
with open("data/names.pkl", "wb") as f:
    pickle.dump(names, f)

print("✅ Saved faces_data.pkl and names.pkl")
print("Sample images were saved to:", samples_dir)
print("Now you can run your attendance script.")
