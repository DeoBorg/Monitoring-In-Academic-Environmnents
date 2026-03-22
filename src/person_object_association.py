from ultralytics import YOLO
import cv2
import os
from person_tracking_bytetrack import ByteTrackPersonTracker


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

    rx1 = int(x1 - 0.35 * w)
    rx2 = int(x2 + 0.35 * w)
    ry1 = int(y1 + 0.20 * h)
    ry2 = int(y2 + 0.90 * h)

    return rx1, ry1, rx2, ry2


def expand_person_region_for_phone(person_bbox):
    x1, y1, x2, y2 = person_bbox
    w = x2 - x1
    h = y2 - y1

    rx1 = int(x1 - 0.15 * w)
    rx2 = int(x2 + 0.15 * w)
    ry1 = int(y1 + 0.10 * h)
    ry2 = int(y2 + 0.85 * h)

    return rx1, ry1, rx2, ry2


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


def main():
    video_path = "data/test_videos/test2.mp4"
    output_path = "outputs/annotated_videos/person_object_association.mp4"

    os.makedirs("outputs/annotated_videos", exist_ok=True)

    tracker = ByteTrackPersonTracker(
        model_path="yolov8n.pt",
        person_conf_threshold=0.25,
        imgsz=1280,
        tracker_config="bytetrack.yaml",
    )

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

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

    frame_count = 0
    person_memory = {}

    object_conf_threshold = 0.12
    retain_frames = 15

    print("Starting improved person-object association...")
    print("Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated = frame.copy()

        # 1) Get tracked persons from reusable ByteTrack module
        tracked_people = tracker.track(frame)

        # 2) Detect laptop + phone on the same frame
        detect_results = model(
            frame,
            verbose=False,
            classes=[63, 67],  # laptop, cell phone
            conf=object_conf_threshold,
            imgsz=1280
        )
        detect_result = detect_results[0]

        laptops = []
        phones = []

        # Collect laptops and phones
        if detect_result.boxes is not None:
            for box in detect_result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])

                if confidence < object_conf_threshold:
                    continue

                if class_name not in {"laptop", "cell phone"}:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                item = {
                    "class_name": class_name,
                    "bbox": (x1, y1, x2, y2),
                    "center": get_box_center(x1, y1, x2, y2),
                    "confidence": confidence
                }

                if class_name == "laptop":
                    laptops.append(item)
                else:
                    phones.append(item)

        # Draw raw laptop and phone detections
        for obj in laptops + phones:
            x1, y1, x2, y2 = obj["bbox"]
            label = f'{obj["class_name"]} {obj["confidence"]:.2f}'

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

        # 3) Associate objects to tracked persons
        for person in tracked_people:
            px1, py1, px2, py2 = person["bbox"]
            person_id = person["id"]

            laptop_region = expand_person_region_for_laptop(person["bbox"])
            phone_region = expand_person_region_for_phone(person["bbox"])

            laptop_region = clip_box_to_frame(laptop_region, width, height)
            phone_region = clip_box_to_frame(phone_region, width, height)

            lx1, ly1, lx2, ly2 = laptop_region
            phx1, phy1, phx2, phy2 = phone_region

            # Debug regions
            cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), (255, 255, 0), 1)
            cv2.rectangle(annotated, (phx1, phy1), (phx2, phy2), (255, 0, 255), 1)

            associated_laptop = None
            associated_phone = None

            # Laptop association
            laptop_candidates = []
            for laptop in laptops:
                center_inside = point_in_box(laptop["center"], laptop_region)
                overlap = iou(laptop["bbox"], laptop_region)
                dist = distance(person["center"], laptop["center"])

                person_w = max(1, px2 - px1)
                person_h = max(1, py2 - py1)
                max_dist = 1.8 * max(person_w, person_h)

                if center_inside or overlap > 0.01 or dist <= max_dist:
                    score = (2.0 * overlap) + (0.6 * laptop["confidence"]) - (0.3 * (dist / max_dist))
                    laptop_candidates.append((score, laptop))

            if laptop_candidates:
                associated_laptop = max(laptop_candidates, key=lambda x: x[0])[1]

            # Phone association
            phone_candidates = []
            for phone in phones:
                center_inside = point_in_box(phone["center"], phone_region)
                overlap = iou(phone["bbox"], phone_region)
                dist = distance(person["center"], phone["center"])

                person_w = max(1, px2 - px1)
                person_h = max(1, py2 - py1)
                max_dist = 1.5 * max(person_w, person_h)

                if center_inside or overlap > 0.01 or dist <= max_dist:
                    score = (1.8 * overlap) + (0.9 * phone["confidence"]) - (0.25 * (dist / max_dist))
                    phone_candidates.append((score, phone))

            if phone_candidates:
                associated_phone = max(phone_candidates, key=lambda x: x[0])[1]

            # Temporal smoothing
            if person_id not in person_memory:
                person_memory[person_id] = {
                    "laptop_last_seen": -10000,
                    "phone_last_seen": -10000
                }

            if associated_laptop:
                person_memory[person_id]["laptop_last_seen"] = frame_count

            if associated_phone:
                person_memory[person_id]["phone_last_seen"] = frame_count

            laptop_recent = frame_count - person_memory[person_id]["laptop_last_seen"] <= retain_frames
            phone_recent = frame_count - person_memory[person_id]["phone_last_seen"] <= retain_frames

            laptop_status = "yes" if (associated_laptop or laptop_recent) else "no"
            phone_status = "yes" if (associated_phone or phone_recent) else "no"

            # Draw tracked person
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)

            cv2.putText(
                annotated,
                f"ID {person_id} person",
                (px1, max(py1 - 35, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
                f"Laptop: {laptop_status} | Phone: {phone_status}",
                (px1, max(py1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

            if associated_laptop:
                cv2.line(annotated, person["center"], associated_laptop["center"], (255, 255, 0), 2)

            if associated_phone:
                cv2.line(annotated, person["center"], associated_phone["center"], (255, 0, 255), 2)

            print(
                f"Frame {frame_count} | ID {person_id} | "
                f"Laptop: {laptop_status} | Phone: {phone_status}"
            )

        cv2.putText(
            annotated,
            f"Frame: {frame_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        out.write(annotated)
        cv2.imshow("Improved Person-Object Association", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Done. Saved to: {output_path}")


if __name__ == "__main__":
    main()