import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import json

output_json_path = "output.json"  # JSON file to store results

if processed_path:
    # Construct JSON data
    detection_result = {
        "type": result if check == "Either Screw or Nut" else "Nut",
        "width": round(size, 2),
        "length": round(length, 2) if check == "Either Screw or Nut" else None,
        "image_path": save_path  # Use the captured image path
    }

    # Save to JSON file
    with open(output_json_path, "w") as json_file:
        json.dump(detection_result, json_file, indent=4)

    print(f"✅ JSON updated: {output_json_path}")


# Set the path to save the captured image in the user's Pictures folder
save_directory = r"C:\\CV\\CV88_camera\\Capture"
processed_directory = r"C:\\CV\\CV88_camera\\process"
save_path = os.path.join(save_directory, "captured_image.jpg")
width0, height0 = 600, 600

def check_nut(RGB, rect):
    image = prepare_image(RGB, rect)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
              param1=50, param2=30, minRadius=0, maxRadius=0)
    difference = abs(gray.shape[0] - gray.shape[1])
    if circles is not None and difference < 50:
        return "Nut"
    else:
        return "Either Screw or Nut"

def find_dimension(width, height):
  pixel = 0.125
  if width < height:
    size = ((width * pixel) / 2) + 1
    length = height * pixel - 2
  else:
    size = ((height * pixel) / 2) + 1
    length = width * pixel - 2
  return size, length

def find_dimension_nut(width, height):
  pixel = 0.125
  diameter1 = ((width * pixel) / 2) + 1
  diameter2 = ((height * pixel) / 2) + 1
  return (diameter1 + diameter2)/2

def classify_screw_or_bolt(img, rect):
    image = prepare_image(img, rect)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,gray = cv2.threshold(gray,130,255,cv2.THRESH_BINARY_INV)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    class_type = 0

    if contours:
        # Get the bottom-most part of the contour (tip of the screw/bolt)
        cnt = contours[0]
        white_pixels = np.column_stack(np.where(edges == 255))

        # Find the pixel with the maximum y value (lowest point)
        lowest_white_pixel = tuple(white_pixels[white_pixels[:, 0].argmax()][::-1])

        print(f"Lowest white pixel (x, y): {lowest_white_pixel}")

        # Extract a small region around the tip for analysis
        x, y = lowest_white_pixel
        region = gray[gray.shape[0]-20:gray.shape[0], 0:gray.shape[1]]

        # Compute sharpness by detecting edges in this small region
        tip_edges = cv2.Canny(region, 100, 200)
        edge_count = np.count_nonzero(tip_edges)  # Count nonzero (edge) pixels

        # Threshold to classify as screw or bolt
        if edge_count < 50:  # More edges = sharper tip (screw)
            label = "Screw"
        else:  # Fewer edges = smooth tip (bolt)
            label = "Bolt"

        # Draw result
        cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
        cv2.putText(image, label, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.circle(image, lowest_white_pixel, 5, (0, 0, 255), -1)
        return label
    else:
        return "No object detected"
    
def prepare_image(RGB, rect):
  box = cv2.boxPoints(rect)  # Get corner points of the bounding box
  box = np.intp(box)  # Convert to integer coordinates
  w = int(rect[1][0])
  h = int(rect[1][1])
  ori = np.float32([box[1], box[2], box[0], box[3]])
  tar = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
  M = cv2.getPerspectiveTransform(ori, tar)
  dst = cv2.warpPerspective(RGB, M, (w, h))
  if dst.shape[0] < dst.shape[1]:
    dst = cv2.rotate(dst, cv2.ROTATE_90_CLOCKWISE)

  check_img = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
  ret,bw2 = cv2.threshold(check_img,170,255,cv2.THRESH_BINARY_INV)
  cropped_image_1 = bw2[0:int(dst.shape[0]/2), 0:int(dst.shape[1])]
  cropped_image_2 = bw2[int(dst.shape[0]/2):int(dst.shape[0]), 0:int(dst.shape[1])]
  if np.sum(cropped_image_1 == 255) < np.sum(cropped_image_2 == 255):
    dst = cv2.rotate(dst, cv2.ROTATE_180)

  # Scale Up Image
  canvas_height, canvas_width = dst.shape[0] + 10, dst.shape[1] + 10
  transparent_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8)*255
  img_height, img_width = dst.shape[:2]
  img_height, img_width = dst.shape[:2]
  x_offset = (canvas_width - img_width) // 2
  y_offset = (canvas_height - img_height) // 2
  transparent_canvas[y_offset : y_offset + img_height, x_offset : x_offset + img_width] = dst

  return transparent_canvas

def draw_rotated_text(image, text, center, angle, 
                      font=cv2.FONT_HERSHEY_SIMPLEX, 
                      font_scale=1, 
                      color=(0, 255, 0), 
                      thickness=1, 
                      padding=30):
    """
    Draw rotated text on an image.

    :param image: Original image (numpy array)
    :param text: Text to draw
    :param center: (x, y) center point for the text
    :param angle: Rotation angle in degrees
    :param font: OpenCV font
    :param font_scale: Font size
    :param color: BGR color tuple
    :param thickness: Text thickness
    :param padding: Extra padding around text to prevent clipping when rotated
    """

    # Get text size (width, height)
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Create larger transparent image (with padding to account for rotation)
    text_img_height = text_size[1] + height0
    text_img_width = text_size[0] + width0
    text_img = np.zeros((text_img_height*2, text_img_width*2, 4), dtype=np.uint8)

    # Draw text in the center of the transparent image
    text_x = (text_img.shape[1] - text_size[0]) // 2
    text_y = (text_img.shape[0] + text_size[1]) // 2

    cv2.putText(text_img, text, (text_x, text_y), font, font_scale, (255, 0, 0, 255), thickness, cv2.LINE_AA)

    # Rotate text image around its center
    rotation_center = (text_img.shape[1] // 2, text_img.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(rotation_center, angle, 1)

    rotated_text_img = cv2.warpAffine(text_img, rotation_matrix, (text_img.shape[1], text_img.shape[0]))

    # Overlay rotated text on main image
    x_offset = int(center[0] - rotated_text_img.shape[1] // 2)
    y_offset = int(center[1] - rotated_text_img.shape[0] // 2)

    for y in range(rotated_text_img.shape[0]):
        for x in range(rotated_text_img.shape[1]):
            if rotated_text_img[y, x][3] > 0:  # Non-transparent pixel
                image_y = y_offset + y
                image_x = x_offset + x
                if 0 <= image_x < image.shape[1] and 0 <= image_y < image.shape[0]:
                    image[image_y, image_x] = rotated_text_img[y, x][:3]

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
    ret, bw1 = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological closing to remove noise
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(bw1, cv2.MORPH_CLOSE, kernel)

    # Find contours
    pixel = 38/152
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

            # Get angle of rotation
            angle = rect[2]
            angle = 90 - angle

            width = int(rect[1][0])
            height = int(rect[1][1])
            if width > height:
                width, height = height, width
            if width < height:
                size = width * pixel
                length = height * pixel
            else:
                size = height * pixel
                length = width * pixel

            check = check_nut(img, rect)

            # Display Cropped Image
            if check == "Either Screw or Nut" :
                size, length = find_dimension(width, height)
                # Print width and height
                print("Bounding Box Width:", size)
                print("Bounding Box Height:", length)
                result = classify_screw_or_bolt(img, rect)
                draw_rotated_text(img, result, box[2], angle)
                print("Detected:", result)
            else :
                size = find_dimension_nut(width, height)
                draw_rotated_text(img, "Nut", box[2], angle)
                print("Diameter of Nut:", size)
            
            textA = str(size) + "," + str(length)
            print("Width: ", size, "Height: ", length)

            # Draw the bounding box
            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
            draw_rotated_text(img, textA, box[0], angle)

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

        test_image = save_path  # Change to a valid image path
        processed_path = process_image(test_image)

        # Update the last capture time
        last_capture_time = current_time

    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()