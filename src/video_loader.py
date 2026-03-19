import cv2

def main():
    video_path = "data/test_videos/test2.mp4"

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return
    
    print("Video opened successfully")
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video or error reading frame")
            break

        cv2.imshow("Video Frame", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("Exiting video playback")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()