import cv2

def capture_image():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Webcam not accessible")
        return

    print("📸 Press SPACE to capture or ESC to exit")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        cv2.imshow("Test Image Capture", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("❌ Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "test.jpg"
            cv2.imwrite(img_name, frame)
            print(f"✅ Saved {img_name} successfully!")
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image()

