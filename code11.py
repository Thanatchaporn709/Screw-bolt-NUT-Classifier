import cv2
import os
import time
import numpy as np

# Set the path to save the captured image in the user's Pictures folder
save_directory = r"E:\Python\CV_Proj\captured"
processed_directory = r"E:\Python\CV_Proj\processed"
save_path = os.path.join(save_directory, "captured_image.jpg")

def process_image(image_path):
    # ✅ Check if the file exists and has a valid extension
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return None

    if not image_path.lower().endswith((".jpg", ".png")):
        print(f"❌ Error: Unsupported file format: {image_path}")
        return None

    print(f"🛠 Processing image: {image_path}")

    # Load the captured image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Failed to load image from {image_path}")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to binary (black & white)
    ret, bw1 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological closing to remove noise
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(bw1, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("❌ Error: No contours found.")
        return None

    if contours:  # Ensure at least one contour is found
        indexObj = 0
        for indexObj in range(len(contours[0])):
            # Get the bounding box for the first contour
            rect = cv2.minAreaRect(contours[0][indexObj])
            box = cv2.boxPoints(rect)
            box = np.intp(box)  # Convert to integer

            # Draw the bounding box
            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
            print("Width: ", int(rect[1][1]), "Height: ", int(rect[1][0]))

        # Save processed image
        processed_filename = f"processed_{os.path.basename(image_path)}"
        processed_image_path = os.path.join(processed_directory, processed_filename)

    success = cv2.imwrite(processed_image_path, img)
    if success:
        print(f"✅ Processed image saved at: {processed_image_path}")
    else:
        print("❌ Error: Failed to save processed image!")

    return processed_image_path

# Open webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Capturing an image every 3 seconds... Press 'q' to quit.")

image_count = 1

# Get the current time for starting the 3-second interval
last_capture_time = time.time()

while True:
    # Capture frame
    ret, frame = cap.read()

    if not ret:
        print("❌ Error: Failed to capture frame!")
        break

    # Display the live webcam feed in a window
    cv2.imshow("Live Webcam Feed", frame)

    # Capture and save image every 3 seconds
    current_time = time.time()

    if current_time - last_capture_time >= 3:  # 3 seconds passed
        timestamp = time.strftime("%Y%m%d-%H%M%S")  # Get a timestamp
        image_filename = f"image_{timestamp}.jpg"  # Create a unique filename
        save_path = os.path.join(save_directory, image_filename)

        success = cv2.imwrite(save_path, frame)  # Save the image

        if success:
            print(f"✅ Image {image_count} saved at: {save_path}")
            image_count += 1  # Increment image count
        else:
            print("❌ Error: Failed to save image!")

        # Update the last capture time
        last_capture_time = current_time

        test_image = save_path  # Change to a valid image path
        processed_path = process_image(test_image)

    # Check fpr 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

