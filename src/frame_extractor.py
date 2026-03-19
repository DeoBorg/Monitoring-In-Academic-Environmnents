import cv2
import os


def main():
    video_path = "data/test_videos/test2.mp4"
    output_folder = "outputs/frames"

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    saved_count = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    SAVE_EVERY = int(fps)  # ~1 frame per second

    print("Starting frame extraction...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % SAVE_EVERY == 0:
            filename = f"frame_{frame_count}.jpg"
            filepath = os.path.join(output_folder, filename)

            cv2.imwrite(filepath, frame)
            saved_count += 1

            print(f"Saved: {filepath}")

    cap.release()

    print("\nDone.")
    print(f"Total frames processed: {frame_count}")
    print(f"Total frames saved: {saved_count}")


if __name__ == "__main__":
    main()