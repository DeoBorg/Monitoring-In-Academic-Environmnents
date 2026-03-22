import os
import sys
from collections import defaultdict, Counter

import cv2
from ultralytics import YOLO


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEDIA_PIPE_DIR = os.path.join(SRC_DIR, "MediaPipe")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
if MEDIA_PIPE_DIR not in sys.path:
    sys.path.append(MEDIA_PIPE_DIR)

from person_tracking_bytetrack import ByteTrackPersonTracker
from person_attention_estimation import (
    PersonAttentionEstimator,
    is_facing_other_person,
    smooth_attention_label,
)


def get_box_center(x1, y1, x2, y2):
    return (x1 + x2) // 2, (y1 + y2) // 2


def point_in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def clip_box_to_frame(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    return x1, y1, x2, y2


def expand_person_region_for_laptop(person_bbox):
    x1, y1, x2, y2 = person_bbox
    w = x2 - x1
    h = y2 - y1
    return int(x1 - 0.35 * w), int(y1 + 0.20 * h), int(x2 + 0.35 * w), int(y2 + 0.90 * h)


def expand_person_region_for_phone(person_bbox):
    x1, y1, x2, y2 = person_bbox
    w = x2 - x1
    h = y2 - y1
    return int(x1 - 0.15 * w), int(y1 + 0.10 * h), int(x2 + 0.15 * w), int(y2 + 0.85 * h)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    union = box_area(box_a) + box_area(box_b) - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def distance(point_a, point_b):
    ax, ay = point_a
    bx, by = point_b
    dx = ax - bx
    dy = ay - by
    return (dx * dx + dy * dy) ** 0.5


def get_dir_candidates(person_state):
    return {
        person_state["attention_smoothed"],
        person_state["face_direction"],
        person_state["body_direction"],
    }


def is_direction_unknown(person_state):
    dirs = get_dir_candidates(person_state)
    known = {"left", "right", "forward", "down"}
    return not any(d in known for d in dirs)


def phone_direction_match(person_state):
    dirs = get_dir_candidates(person_state)
    if "down" in dirs:
        return 1.0

    phone_item = person_state["phone_item"]
    if phone_item is None:
        return 0.0

    person_x = person_state["center"][0]
    phone_x = phone_item["center"][0]
    if phone_x > person_x and "right" in dirs:
        return 1.0
    if phone_x < person_x and "left" in dirs:
        return 1.0
    return 0.0


def laptop_direction_match(person_state):
    dirs = get_dir_candidates(person_state)
    if "down" in dirs:
        return 1.0
    if "forward" in dirs:
        return 0.8
    return 0.0


def laptop_geometry_score(person_state):
    laptop_item = person_state["laptop_item"]
    if laptop_item is None:
        return 0.0

    x1, y1, x2, y2 = person_state["bbox"]
    person_w = max(1, x2 - x1)
    person_h = max(1, y2 - y1)
    person_cx, person_cy = person_state["center"]
    laptop_cx, laptop_cy = laptop_item["center"]

    norm_x = abs(laptop_cx - person_cx) / person_w
    norm_y = (laptop_cy - person_cy) / person_h

    score = 0.0
    # Laptop typically lies below or around torso center.
    if 0.0 <= norm_y <= 0.8:
        score += 0.7
    # Laptop usually not far horizontally from user.
    if norm_x <= 0.8:
        score += 0.5
    # Detection confidence contributes but is capped.
    score += min(0.4, laptop_item["confidence"] * 0.5)
    return score


def phone_geometry_score(person_state):
    phone_item = person_state["phone_item"]
    if phone_item is None:
        return 0.0

    x1, y1, x2, y2 = person_state["bbox"]
    person_w = max(1, x2 - x1)
    person_h = max(1, y2 - y1)
    person_cx, person_cy = person_state["center"]
    phone_cx, phone_cy = phone_item["center"]

    norm_x = abs(phone_cx - person_cx) / person_w
    norm_y = abs(phone_cy - person_cy) / person_h

    score = 0.0
    # Phone should be fairly near torso/hand area.
    if norm_x <= 0.9:
        score += 0.6
    if norm_y <= 0.9:
        score += 0.6
    score += min(0.4, phone_item["confidence"] * 0.5)
    return score


def compute_target_scores(person_state, mutual_facing_ids):
    person_id = person_state["id"]
    direction_unknown = is_direction_unknown(person_state)

    score_person = 0.0
    score_phone = 0.0
    score_laptop = 0.0
    score_away = 0.2

    if person_id in mutual_facing_ids:
        score_person += 2.5

    if person_state["has_phone"]:
        score_phone += 0.8
        score_phone += phone_geometry_score(person_state)
        score_phone += phone_direction_match(person_state)

    if person_state["has_laptop"]:
        score_laptop += 1.0
        score_laptop += laptop_geometry_score(person_state)
        score_laptop += laptop_direction_match(person_state)

    # If no reliable direction is available, avoid over-penalizing object-based targets.
    if direction_unknown:
        score_away -= 0.15
    else:
        dirs = get_dir_candidates(person_state)
        if "left" in dirs or "right" in dirs:
            score_away += 0.3

    return {
        "looking_at_person": score_person,
        "looking_at_phone": score_phone,
        "looking_at_laptop": score_laptop,
        "looking_away": score_away,
    }


def smooth_target_label(target_history, person_id, new_label, window_size=12):
    if person_id not in target_history:
        target_history[person_id] = []
    target_history[person_id].append(new_label)
    if len(target_history[person_id]) > window_size:
        target_history[person_id].pop(0)
    counts = Counter(target_history[person_id])
    return counts.most_common(1)[0][0]


def decide_attention_target(person_state, mutual_facing_ids, previous_target):
    scores = compute_target_scores(person_state, mutual_facing_ids)
    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    # Hysteresis: keep previous label unless new winner is meaningfully stronger.
    if previous_target is not None and previous_target in scores:
        if scores[previous_target] + 0.35 >= best_score:
            return previous_target, scores

    # Safety fallback: if a person has a retained laptop and no phone evidence,
    # classify as laptop-targeted rather than away under ambiguous orientation.
    if (
        best_label == "looking_away"
        and person_state["has_laptop"]
        and not person_state["has_phone"]
    ):
        return "looking_at_laptop", scores

    return best_label, scores


def associate_objects_for_person(person, laptops, phones, width, height):
    px1, py1, px2, py2 = person["bbox"]
    laptop_region = clip_box_to_frame(expand_person_region_for_laptop(person["bbox"]), width, height)
    phone_region = clip_box_to_frame(expand_person_region_for_phone(person["bbox"]), width, height)

    laptop_candidates = []
    for laptop in laptops:
        center_inside = point_in_box(laptop["center"], laptop_region)
        overlap = iou(laptop["bbox"], laptop_region)
        dist = distance(person["center"], laptop["center"])
        max_dist = 1.8 * max(1, px2 - px1, py2 - py1)
        if center_inside or overlap > 0.01 or dist <= max_dist:
            score = (2.0 * overlap) + (0.6 * laptop["confidence"]) - (0.3 * (dist / max_dist))
            laptop_candidates.append((score, laptop))

    phone_candidates = []
    for phone in phones:
        center_inside = point_in_box(phone["center"], phone_region)
        overlap = iou(phone["bbox"], phone_region)
        dist = distance(person["center"], phone["center"])
        max_dist = 1.5 * max(1, px2 - px1, py2 - py1)
        if center_inside or overlap > 0.01 or dist <= max_dist:
            score = (1.8 * overlap) + (0.9 * phone["confidence"]) - (0.25 * (dist / max_dist))
            phone_candidates.append((score, phone))

    associated_laptop = max(laptop_candidates, key=lambda x: x[0])[1] if laptop_candidates else None
    associated_phone = max(phone_candidates, key=lambda x: x[0])[1] if phone_candidates else None
    return associated_laptop, associated_phone, laptop_region, phone_region


def main():
    video_path = "data/test_videos/test2.mp4"
    output_path = "outputs/annotated_videos/attention_object_fusion.mp4"

    os.makedirs("outputs/annotated_videos", exist_ok=True)

    tracker = ByteTrackPersonTracker(
        model_path="yolov8n.pt",
        person_conf_threshold=0.25,
        imgsz=1280,
        tracker_config="bytetrack.yaml",
    )
    object_model = YOLO("yolov8n.pt")
    estimator = PersonAttentionEstimator()

    if estimator.backend == "none":
        print("Warning: MediaPipe direction backend unavailable. Orientation-driven targets may stay conservative.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not create output video at {output_path}")
        cap.release()
        estimator.close()
        return

    frame_count = 0
    object_conf_threshold = 0.12
    retain_frames = 15
    attention_history = defaultdict(list)
    person_memory = {}
    target_history = defaultdict(list)

    print("Running fused attention-object estimation...")
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated = frame.copy()

        tracked_people = tracker.track(frame)

        detect_results = object_model(
            frame,
            verbose=False,
            classes=[63, 67],
            conf=object_conf_threshold,
            imgsz=1280,
        )
        detect_result = detect_results[0]

        laptops = []
        phones = []
        if detect_result.boxes is not None:
            for box in detect_result.boxes:
                class_id = int(box.cls[0])
                class_name = object_model.names[class_id]
                conf = float(box.conf[0])
                if conf < object_conf_threshold or class_name not in {"laptop", "cell phone"}:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                item = {
                    "class_name": class_name,
                    "bbox": (x1, y1, x2, y2),
                    "center": get_box_center(x1, y1, x2, y2),
                    "confidence": conf,
                }
                if class_name == "laptop":
                    laptops.append(item)
                else:
                    phones.append(item)

        person_states = []
        for person in tracked_people:
            person_id = person["id"]
            associated_laptop, associated_phone, laptop_region, phone_region = associate_objects_for_person(
                person, laptops, phones, width, height
            )

            if person_id not in person_memory:
                person_memory[person_id] = {
                    "laptop_last_seen": -10_000,
                    "phone_last_seen": -10_000,
                    "last_laptop_item": None,
                    "last_phone_item": None,
                }

            if associated_laptop is not None:
                person_memory[person_id]["laptop_last_seen"] = frame_count
                person_memory[person_id]["last_laptop_item"] = associated_laptop
            if associated_phone is not None:
                person_memory[person_id]["phone_last_seen"] = frame_count
                person_memory[person_id]["last_phone_item"] = associated_phone

            has_laptop = (
                associated_laptop is not None
                or (frame_count - person_memory[person_id]["laptop_last_seen"] <= retain_frames)
            )
            has_phone = (
                associated_phone is not None
                or (frame_count - person_memory[person_id]["phone_last_seen"] <= retain_frames)
            )

            laptop_item = associated_laptop or person_memory[person_id]["last_laptop_item"]
            phone_item = associated_phone or person_memory[person_id]["last_phone_item"]

            attn = estimator.estimate_attention(frame, person)
            attention_smoothed = smooth_attention_label(
                attention_history, person_id, attn["attention"], window_size=10
            )

            person_states.append(
                {
                    "id": person_id,
                    "bbox": person["bbox"],
                    "center": person["center"],
                    "face_direction": attn["face_direction"],
                    "body_direction": attn["body_direction"],
                    "attention_smoothed": attention_smoothed,
                    "has_laptop": has_laptop,
                    "has_phone": has_phone,
                    "laptop_item": laptop_item,
                    "phone_item": phone_item,
                    "laptop_region": laptop_region,
                    "phone_region": phone_region,
                }
            )

        mutual_facing_ids = set()
        for i in range(len(person_states)):
            for j in range(i + 1, len(person_states)):
                a = person_states[i]
                b = person_states[j]
                if is_facing_other_person(a, b, a["attention_smoothed"], b["attention_smoothed"]):
                    mutual_facing_ids.add(a["id"])
                    mutual_facing_ids.add(b["id"])

        for obj in laptops + phones:
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                annotated,
                f'{obj["class_name"]} {obj["confidence"]:.2f}',
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

        for state in person_states:
            x1, y1, x2, y2 = state["bbox"]
            person_id = state["id"]
            previous_target = person_memory.get(person_id, {}).get("last_target")
            target_raw, target_scores = decide_attention_target(state, mutual_facing_ids, previous_target)
            target = smooth_target_label(target_history, person_id, target_raw, window_size=12)
            person_memory[person_id]["last_target"] = target

            lx1, ly1, lx2, ly2 = state["laptop_region"]
            phx1, phy1, phx2, phy2 = state["phone_region"]
            cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), (255, 255, 0), 1)
            cv2.rectangle(annotated, (phx1, phy1), (phx2, phy2), (255, 0, 255), 1)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            line_1 = (
                f"ID {person_id} | attn:{state['attention_smoothed']} | "
                f"ph:{'yes' if state['has_phone'] else 'no'} | lp:{'yes' if state['has_laptop'] else 'no'}"
            )
            line_2 = f"target:{target}"
            line_3 = f"scores p:{target_scores['looking_at_person']:.1f} ph:{target_scores['looking_at_phone']:.1f} lp:{target_scores['looking_at_laptop']:.1f}"

            cv2.putText(
                annotated,
                line_1,
                (x1, max(20, y1 - 28)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated,
                line_2,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                annotated,
                line_3,
                (x1, max(20, y1 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 200, 0),
                1,
            )

        cv2.putText(
            annotated,
            f"Frame: {frame_count}/{total_frames}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )

        out.write(annotated)
        cv2.imshow("Attention Object Fusion", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    estimator.close()
    cv2.destroyAllWindows()
    print(f"Done. Saved to: {output_path}")


if __name__ == "__main__":
    main()
