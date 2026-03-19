from ultralytics import YOLO
import cv2
import os


def main():
    video_path = "data/test_videos/test2.mp4"
    output_path = "outputs/annotated_videos/annotated_test2.mp4"

    os.makedirs("outputs/annotated_videos", exist_ok=True)

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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not create output video at {output_path}")
        cap.release()
        return

    print("Processing video...")
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    print(f"Total frames: {total_frames}")

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        results = model(frame, verbose=False)
        result = results[0]

        detection_count = 0

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            if class_name not in target_classes or confidence < confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{class_name} {confidence:.2f}"
            detection_count += 1

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

        cv2.putText(
            frame,
            f"Detections: {detection_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        out.write(frame)

        cv2.imshow("Filtered YOLO Video Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopped by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Done.")
    print(f"Saved annotated video to: {output_path}")


if __name__ == "__main__":
    main()