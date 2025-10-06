from fer import FER
import cv2
import traceback

try:
    # Load image
    img = cv2.imread("test.jpg")
    if img is None:
        print("❌ Could not read image. Make sure 'test.jpg' exists in this folder.")
        exit()

    print("ℹ️ Image loaded — shape:", img.shape)
    print("ℹ️ Creating FER detector with mtcnn=True ...")

    detector = FER(mtcnn=True)
    print("✅ Detector created successfully.")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_emotions(rgb)
    print("✅ detect_emotions() returned:", results)

except Exception as e:
    print("❌ Exception occurred:")
    traceback.print_exc()
