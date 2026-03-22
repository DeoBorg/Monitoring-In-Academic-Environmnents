from ultralytics import YOLO
import cv2


class ByteTrackPersonTracker:
    def __init__(
        self,
        model_path="yolov8n.pt",
        person_conf_threshold=0.25,
        imgsz=1280,
        tracker_config="bytetrack.yaml",
    ):
        self.model = YOLO(model_path)
        self.person_conf_threshold = person_conf_threshold
        self.imgsz = imgsz
        self.tracker_config = tracker_config

    @staticmethod
    def get_box_center(x1, y1, x2, y2):
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def track(self, frame):
        """
        Tracks persons in a single frame using YOLO + ByteTrack.

        Returns:
            tracked_people: list of dicts like
            [
                {
                    "id": 1,
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "confidence": 0.91
                }
            ]
        """
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            classes=[0],  # COCO class 0 = person
            conf=self.person_conf_threshold,
            imgsz=self.imgsz,
            tracker=self.tracker_config,
        )

        tracked_people = []

        if not results or results[0].boxes is None:
            return tracked_people

        for box in results[0].boxes:
            if box.id is None:
                continue

            confidence = float(box.conf[0])
            if confidence < self.person_conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            tracked_people.append({
                "id": int(box.id[0]),
                "bbox": (x1, y1, x2, y2),
                "center": self.get_box_center(x1, y1, x2, y2),
                "confidence": confidence,
            })

        return tracked_people


def draw_tracked_people(frame, tracked_people):
    annotated = frame.copy()

    for person in tracked_people:
        x1, y1, x2, y2 = person["bbox"]
        person_id = person["id"]
        confidence = person["confidence"]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"ByteTrack ID {person_id} person {confidence:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return annotated


def main():
    video_path = "data/test_videos/test2.mp4"

    tracker = ByteTrackPersonTracker(
        model_path="yolov8n.pt",
        person_conf_threshold=0.25,
        imgsz=1280,
        tracker_config="bytetrack.yaml",
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    print("Running reusable ByteTrack person tracker...")
    print("Press 'q' to stop.")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        tracked_people = tracker.track(frame)
        annotated = draw_tracked_people(frame, tracked_people)

        cv2.putText(
            annotated,
            f"Frame: {frame_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        cv2.imshow("Reusable ByteTrack Person Tracker", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()