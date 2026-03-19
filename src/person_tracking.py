from ultralytics import YOLO
import cv2
import os


def main():
    video_path = "data/test_videos/test2.mp4"
    output_path = "outputs/annotated_videos/tracked_test2.mp4"

    os.makedirs("outputs/annotated_videos", exist_ok=True)

    model = YOLO("yolov8n.pt")

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

    print("Starting person tracking...")
    print(f"Input video: {video_path}")
    print(f"Output video: {output_path}")
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    print(f"Total frames: {total_frames}")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = model.track(
            frame,
            persist=True,
            verbose=False,
            classes=[0]  # class 0 = person in COCO
        )

        annotated_frame = frame.copy()

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                if box.id is None:
                    continue

                track_id = int(box.id[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = f"ID {track_id} person {confidence:.2f}"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        cv2.putText(
            annotated_frame,
            f"Frame: {frame_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        out.write(annotated_frame)
        cv2.imshow("Person Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopped by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Done.")
    print(f"Tracked video saved to: {output_path}")


if __name__ == "__main__":
    main()