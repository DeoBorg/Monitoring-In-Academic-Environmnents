from ultralytics import YOLO
import cv2


def main():
    image_path = "outputs/frames/frame_116.jpg"

    model = YOLO("yolov8n.pt")

    results = model(image_path)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Detection", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Detection complete.")


if __name__ == "__main__":
    main()