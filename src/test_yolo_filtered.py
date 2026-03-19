from ultralytics import YOLO
import cv2


def main():
    image_path = "outputs/frames/frame_30.jpg"

    model = YOLO("yolov8n.pt")
    results = model(image_path)

    result = results[0]
    image = cv2.imread(image_path)

    target_classes = {"person", "cell phone", "laptop"}

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        if class_name not in target_classes:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        print(f"Detected: {class_name} ({confidence:.2f}) at [{x1}, {y1}, {x2}, {y2}]")

    cv2.imshow("Filtered YOLO Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()