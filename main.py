import math
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def ensure_hand_model() -> Path:
    model_path = Path(__file__).with_name("hand_landmarker.task")
    if model_path.exists():
        return model_path

    model_url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    print("Downloading hand model...")
    urlretrieve(model_url, model_path)
    return model_path


def to_pixel(landmark, width: int, height: int) -> tuple[int, int]:
    return int(landmark.x * width), int(landmark.y * height)


def mirror_handedness(label: str) -> str:
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


def draw_hand(frame, landmarks, width: int, height: int) -> None:
    for start, end in HAND_CONNECTIONS:
        x1, y1 = to_pixel(landmarks[start], width, height)
        x2, y2 = to_pixel(landmarks[end], width, height)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for lm in landmarks:
        x, y = to_pixel(lm, width, height)
        cv2.circle(frame, (x, y), 4, (255, 100, 0), -1)


def main() -> None:
    model_path = ensure_hand_model()

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    with vision.HandLandmarker.create_from_options(options) as hand_landmarker:
        frame_index = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            results = hand_landmarker.detect_for_video(mp_image, frame_index * 33)
            frame_index += 1

            hand_landmarks = getattr(results, "hand_landmarks", [])
            handedness_info = getattr(results, "handedness", [])

            if hand_landmarks and handedness_info:
                index_finger_positions = []

                for idx, landmarks in enumerate(hand_landmarks):
                    draw_hand(frame, landmarks, w, h)

                    label = "Unknown"
                    if idx < len(handedness_info) and handedness_info[idx]:
                        label = handedness_info[idx][0].category_name
                    label = mirror_handedness(label)

                    ix, iy = to_pixel(landmarks[8], w, h)
                    index_finger_positions.append((ix, iy))

                    wx, wy = to_pixel(landmarks[0], w, h)
                    cv2.putText(
                        frame,
                        label,
                        (wx, wy + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 100, 0),
                        2,
                    )

                if len(index_finger_positions) == 2:
                    pt1 = index_finger_positions[0]
                    pt2 = index_finger_positions[1]

                    distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

                    if distance < 50:
                        cv2.putText(
                            frame,
                            "FINGERS TOUCHING!",
                            (w // 2 - 150, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 0, 255),
                            3,
                        )

            cv2.imshow("Two-Hand Interaction", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()