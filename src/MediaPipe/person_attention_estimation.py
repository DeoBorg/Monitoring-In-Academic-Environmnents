import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import math
import mediapipe as mp
import time
from collections import defaultdict, Counter

from person_tracking_bytetrack import ByteTrackPersonTracker


def clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    return x1, y1, x2, y2


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def angle_between_points(a, b):
    ax, ay = a
    bx, by = b
    return math.degrees(math.atan2(by - ay, bx - ax))


def distance(a, b):
    ax, ay = a
    bx, by = b
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


class PersonAttentionEstimator:
    def __init__(self):
        self.backend = "none"
        self.mp_face = None
        self.mp_pose = None
        self.face_mesh = None
        self.pose = None

        if hasattr(mp, "solutions"):
            self.backend = "solutions"
            self.mp_face = mp.solutions.face_mesh
            self.mp_pose = mp.solutions.pose

            self.face_mesh = self.mp_face.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            # Python 3.12 environments may install a Tasks-only mediapipe package.
            # Keep pipeline running and fail-soft with a clear explanation.
            print(
                "Warning: this mediapipe build has no mp.solutions API. "
                "Attention will default to 'unknown' until you install a solutions-capable build "
                "(commonly Python 3.11 + mediapipe 0.10.x) or implement tasks model assets."
            )

    def estimate_face_direction(self, crop_bgr):
        """
        Returns one of:
        - left
        - right
        - forward
        - down
        - unknown
        """
        if self.backend != "solutions" or self.face_mesh is None:
            return "unknown", None

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(crop_rgb)

        if not result.multi_face_landmarks:
            return "unknown", None

        landmarks = result.multi_face_landmarks[0].landmark
        h, w, _ = crop_bgr.shape

        # Approximate key landmarks from MediaPipe Face Mesh
        nose_tip = landmarks[1]
        left_eye_outer = landmarks[33]
        right_eye_outer = landmarks[263]
        chin = landmarks[152]
        forehead = landmarks[10]

        nose_px = (int(nose_tip.x * w), int(nose_tip.y * h))
        left_eye_px = (int(left_eye_outer.x * w), int(left_eye_outer.y * h))
        right_eye_px = (int(right_eye_outer.x * w), int(right_eye_outer.y * h))
        chin_px = (int(chin.x * w), int(chin.y * h))
        forehead_px = (int(forehead.x * w), int(forehead.y * h))

        face_center_x = (left_eye_px[0] + right_eye_px[0]) / 2
        eye_width = max(1, abs(right_eye_px[0] - left_eye_px[0]))
        nose_offset = (nose_px[0] - face_center_x) / eye_width

        face_height = max(1, abs(chin_px[1] - forehead_px[1]))
        nose_vertical_ratio = (nose_px[1] - forehead_px[1]) / face_height

        # Heuristics
        if nose_vertical_ratio > 0.62:
            direction = "down"
        elif nose_offset < -0.12:
            direction = "left"
        elif nose_offset > 0.12:
            direction = "right"
        else:
            direction = "forward"

        debug = {
            "nose": nose_px,
            "left_eye": left_eye_px,
            "right_eye": right_eye_px,
            "chin": chin_px,
            "forehead": forehead_px,
            "nose_offset": nose_offset,
            "nose_vertical_ratio": nose_vertical_ratio,
        }
        return direction, debug

    def estimate_body_direction(self, crop_bgr):
        """
        Returns:
        - left
        - right
        - forward
        - unknown
        """
        if self.backend != "solutions" or self.pose is None:
            return "unknown", None

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        result = self.pose.process(crop_rgb)

        if not result.pose_landmarks:
            return "unknown", None

        landmarks = result.pose_landmarks.landmark
        h, w, _ = crop_bgr.shape

        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]

        ls = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        rs = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        ns = (int(nose.x * w), int(nose.y * h))

        shoulder_center_x = (ls[0] + rs[0]) / 2
        shoulder_width = max(1, abs(rs[0] - ls[0]))
        nose_offset = (ns[0] - shoulder_center_x) / shoulder_width

        if nose_offset < -0.15:
            direction = "left"
        elif nose_offset > 0.15:
            direction = "right"
        else:
            direction = "forward"

        debug = {
            "left_shoulder": ls,
            "right_shoulder": rs,
            "nose": ns,
            "nose_offset": nose_offset,
        }
        return direction, debug

    def estimate_attention(self, frame, person):
        x1, y1, x2, y2 = person["bbox"]
        h, w, _ = frame.shape
        x1, y1, x2, y2 = clamp_box((x1, y1, x2, y2), w, h)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or (x2 - x1) < 40 or (y2 - y1) < 40:
            return {
                "id": person["id"],
                "bbox": person["bbox"],
                "center": person["center"],
                "face_direction": "unknown",
                "body_direction": "unknown",
                "attention": "unknown"
            }

        face_direction, face_debug = self.estimate_face_direction(crop)
        body_direction, body_debug = self.estimate_body_direction(crop)

        # Simple priority:
        # face direction if available, otherwise body direction
        if face_direction != "unknown":
            attention = face_direction
        else:
            attention = body_direction

        return {
            "id": person["id"],
            "bbox": person["bbox"],
            "center": person["center"],
            "face_direction": face_direction,
            "body_direction": body_direction,
            "attention": attention,
            "face_debug": face_debug,
            "body_debug": body_debug,
        }

    def close(self):
        if self.face_mesh is not None:
            self.face_mesh.close()
        if self.pose is not None:
            self.pose.close()


def is_facing_other_person(person_a, person_b, direction_a, direction_b):
    ax, ay = person_a["center"]
    bx, by = person_b["center"]

    horizontal_gap = bx - ax

    a_faces_b = (
        (horizontal_gap > 0 and direction_a == "right") or
        (horizontal_gap < 0 and direction_a == "left")
    )

    b_faces_a = (
        (horizontal_gap > 0 and direction_b == "left") or
        (horizontal_gap < 0 and direction_b == "right")
    )

    return a_faces_b and b_faces_a 


def smooth_attention_label(attention_history, person_id, new_label, window_size=10):
    if person_id not in attention_history:
        attention_history[person_id] = []

    if new_label != "unknown":
        attention_history[person_id].append(new_label)
        if len(attention_history[person_id]) > window_size:
            attention_history[person_id].pop(0)

    history = attention_history[person_id]
    if not history:
        return new_label

    counts = Counter(history)
    return counts.most_common(1)[0][0]


def main():
    video_path = "data/test_videos/test2.mp4"

    tracker = ByteTrackPersonTracker(
        model_path="yolov8n.pt",
        person_conf_threshold=0.25,
        imgsz=1280,
        tracker_config="bytetrack.yaml",
    )

    estimator = PersonAttentionEstimator()
    attention_history = defaultdict(list)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open {video_path}")
        return
    
    output_path = "outputs/annotated_videos/person_attention_estimation.mp4"
    os.makedirs("outputs/annotated_videos", exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not create output video at {output_path}")
        cap.release()
        return 

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    job_start = time.perf_counter()

    print("Running simple MediaPipe attention estimation...")
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    print(f"FPS: {fps:.2f} | Frames: {total_frames}")
    print("Press q to quit.")

    while True:
        frame_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        tracked_people = tracker.track(frame)
        attention_results = []

        for person in tracked_people:
            result = estimator.estimate_attention(frame, person)
            result["attention_smoothed"] = smooth_attention_label(
                attention_history,
                result["id"],
                result["attention"],
                window_size=10
            )
            attention_results.append(result)

        # Mutual face-to-face check
        chatting_pairs = set()
        for i in range(len(attention_results)):
            for j in range(i + 1, len(attention_results)):
                a = attention_results[i]
                b = attention_results[j]

                if is_facing_other_person(a, b, a["attention_smoothed"], b["attention_smoothed"]):
                    chatting_pairs.add((a["id"], b["id"]))

        annotated = frame.copy()

        for result in attention_results:
            x1, y1, x2, y2 = result["bbox"]
            person_id = result["id"]

            label = (
                f"ID {person_id} | face:{result['face_direction']} | "
                f"body:{result['body_direction']} | attn:{result['attention_smoothed']}"
            )

            is_chatting = any(person_id in pair for pair in chatting_pairs)
            if is_chatting:
                label += " | mutual-facing"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        cv2.putText(
            annotated,
            f"Frame {frame_count}/{total_frames}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        out.write(annotated)
        cv2.imshow("Person Attention Estimation", annotated)

        # Keep playback close to source FPS instead of racing through frames.
        target_frame_time = 1.0 / fps
        frame_elapsed = time.perf_counter() - frame_start
        delay_ms = max(1, int((target_frame_time - frame_elapsed) * 1000))

        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    estimator.close()

    total_elapsed = time.perf_counter() - job_start
    print(f"Processed frames: {frame_count}/{total_frames}")
    print(f"Elapsed: {total_elapsed:.2f}s")
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()
