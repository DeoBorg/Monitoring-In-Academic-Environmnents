from ultralytics import YOLO
import cv2


def main():
    video_path = "data/test_videos/test1.mp4"

    model = YOLO("yolov8n.pt")
    target_classes = {"person", "cell phone", "laptop"}
    confidence_threshold = 0.40

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Video opened successfully.")
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    print("Press 'q' to quit.")

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video reached.")
            break

        frame_count += 1

        results = model(frame, verbose=False)
        result = results[0]

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            if class_name not in target_classes or confidence < confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{class_name} {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Frame: {frame_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        cv2.imshow("Filtered YOLO Video Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Video stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()