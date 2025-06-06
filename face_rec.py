# imports
import cv2
from deepface import DeepFace
print(DeepFace.__version__)
import pickle
from numpy import dot
from numpy.linalg import norm
from sense_hat import SenseHat
import time

sense = SenseHat()

# Sense Hat colors
W = [255, 255, 255]  # White
R = [255, 0, 0]      # Red
G = [0, 255, 0]      # Green
B = [0, 0, 255]      # Blue
Y = [255, 255, 0]    # Yellow
P = [191, 64, 191]   # Purple
_ = [0, 0, 0]        # Off

# LED patterns
def make_circle(hatColor):
    _ = [0, 0, 0]
    C = hatColor
    return [
        _, _, C, C, C, C, _, _,
        _, C, _, _, _, _, C, _,
        C, _, _, _, _, _, _, C,
        C, _, _, _, _, _, _, C,
        C, _, _, _, _, _, _, C,
        C, _, _, _, _, _, _, C,
        _, C, _, _, _, _, C, _,
        _, _, C, C, C, C, _, _
    ]

def make_smiley(hatColor):
    _ = [0, 0, 0]
    C = hatColor
    return [
        _, _, C, C, C, C, _, _,
        _, C, _, _, _, _, C, _,
        C, _, C, _, _, C, _, C,
        C, _, _, _, _, _, _, C,
        C, C, _, _, _, C, _, C,
        C, _, C, C, C, _, _, C,
        _, C, _, _, _, _, C, _,
        _, _, C, C, C, C, _, _
    ]

def make_face(hatColor):
    _ = [0, 0, 0]
    C = hatColor
    return [
        _, _, _, C, C, _, _, _,
        _, _, C, _, _, C, _, _,
        _, C, _, _, _, _, C, _,
        _, C, _, _, _, _, C, _,
        _, _, C, _, _, C, _, _,
        _, _, _, _, _, _, _, _,
        _, C, C, C, C, C, C, _,
        C, _, _, _, _, _, _, C
    ]



# ---------- Enrollment Mode ----------
print("Starting Enrollment Mode")
circle = make_circle(W)
sense.set_pixels(circle)

# Webcam Capture
max_users = 2
current_id = 1
known_faces = []

while current_id <= max_users:
    cam = cv2.VideoCapture(0)
    cv2.namedWindow(f"Enrollment - User {current_id} (Press joystick or ESC)")

    circle = make_circle(B)
    sense.set_pixels(circle)

    exit_requested = False
    latest_frame = None

    def handle_joystick_enroll(event):
        global exit_requested
        if event.action == "pressed" and event.direction == "middle":
            exit_requested = True

    sense.stick.direction_middle = handle_joystick_enroll

    while not exit_requested:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame.")
            break
        latest_frame = frame
        cv2.imshow(f"Enrollment - User {current_id}", frame)
        key = cv2.waitKey(1)
        if key == 27:
            exit_requested = True
        time.sleep(0.01)

    cam.release()
    cv2.destroyAllWindows()

    if latest_frame is not None:
        filename = f"enroll_face_{current_id}.jpg"
        latest_frame = cv2.cvtColor(latest_frame,cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, latest_frame)
        print("Captured image size:",latest_frame.shape)
        print(f"Saved {filename}")
        
        profile_face = make_face(P)
        sense.set_pixels(profile_face)
        
        print("Generating embedding...")
        embedding_obj = DeepFace.represent(img_path=filename, model_name="Facenet512", enforce_detection=True)
        embedding = embedding_obj[0]["embedding"]

        
        known_faces.append({"id": f"user_{current_id}", "embedding": embedding})
        print(f"user_{current_id} enrolled.")

    current_id += 1

# Save all embeddings
with open("known_faces.pkl", "wb") as f:
    pickle.dump(known_faces, f)

print("All embeddings saved to known_faces.pkl")

smiley = make_smiley(Y)
sense.set_pixels(smiley)

# ---------- Live Recognition Mode ----------
print("\nSwitching to Live Continuous Recognition Mode")
circle = make_circle(B)
sense.set_pixels(circle)

cam = cv2.VideoCapture(0)
cv2.namedWindow("Live Recognition (ESC to exit)")

with open("known_faces.pkl", "rb") as f:
    known_faces = pickle.load(f)

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

last_recognition_time = time.time()

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert to RGB for DeepFace
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Save resized temporary frame
        resized_frame = cv2.resize(frame_rgb, (320, 240))
        cv2.imwrite("live_frame.jpg", resized_frame)

        current_time = time.time()
        if current_time - last_recognition_time >= 2:
            try:
                embedding_obj = DeepFace.represent(
                    img_path="live_frame.jpg",
                    model_name="Facenet512",
                    enforce_detection=False
                )
                test_embedding = embedding_obj[0]["embedding"]

                best_match = None
                highest_similarity = -1

                for person in known_faces:
                    similarity = cosine_similarity(test_embedding, person["embedding"])
                    print(f"Comparing with {person['id']}: {similarity:.4f}")
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = person["id"]

                threshold = 0.6
                if highest_similarity >= threshold:
                    print(f"Match: {best_match} ({highest_similarity:.4f})")
                    circle = make_circle(G)
                    sense.set_pixels(circle)
                    cv2.putText(frame, f"Match: {best_match}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    print(f"Unknown face ({highest_similarity:.4f})")
                    circle = make_circle(R)
                    sense.set_pixels(circle)
                    cv2.putText(frame, "Unknown Face", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                last_recognition_time = current_time

            except Exception as e:
                print("No face detected or recognition error:", e)

        # Show the live feed with overlay
        cv2.imshow("Live Recognition (ESC to exit)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cam.release()
    cv2.destroyAllWindows()
    sense.clear()
    print("Live Recognition ended.")


