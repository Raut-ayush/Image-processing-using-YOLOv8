#code to detect, count and save object images with real timestamps
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import os
from datetime import datetime  # Import datetime module

# Avoid potential issues with duplicate libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Set the confidence threshold to 90%
model.conf = 0.9

# Initialize video capture (0 for the first webcam connected or provide video path)
video_path = "Highway02.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception(f"Error reading video file {video_path}")

# Get video properties
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points to cover the entire frame
region_points = [(0, 0), (w, 0), (w, h), (0, h)]

# Ensure output directory exists
output_dir = "optxt/refrence7"
os.makedirs(output_dir, exist_ok=True)

# Initialize video writer
output_path = os.path.join(output_dir, "refrence7.avi")
video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# Initialize Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(
    view_img=False,
    reg_pts=region_points,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=2
)

# Initialize dictionary to count unique objects
object_counts = {}
unique_objects = set()

# Frame counter
frame_count = 0

# Open a file to log detection times
log_path = os.path.join(output_dir, "detection_times7.txt")
with open(log_path, "w") as log_file:
    log_file.write("Object Detection Log:\n")
    log_file.write("Frame Number, Timestamp, Object Name\n")

    # Start processing the video frames
    try:
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            frame_count += 1  # Increment frame count

            # Calculate the timestamp for the current frame
            timestamp = frame_count / fps

            # Get the current date and time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Perform object tracking on the frame
            results = model.track(im0, persist=True, show=False)

            # Count objects in the current frame
            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls)  # Get the class ID
                    object_name = model.names[class_id]
                    object_id = int(box.id)  # Get the unique object ID

                    if object_id not in unique_objects:
                        unique_objects.add(object_id)

                        if object_name not in object_counts:
                            object_counts[object_name] = 0  # Initialize if not present

                        object_counts[object_name] += 1

                        # Save the frame where a new unique object appears
                        margin = 20  # Margin around the detected object
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert tensor to list of integers
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(w, x2 + margin)
                        y2 = min(h, y2 + margin)
                        roi = im0[y1:y2, x1:x2]
                        new_object_frame_path = os.path.join(output_dir, f"new_object_{object_name}_{frame_count}.jpg")
                        cv2.imwrite(new_object_frame_path, roi)
                        print(f"New unique object '{object_name}' detected and saved at frame {frame_count}, time {timestamp:.2f}s, date {current_time}.")

                        # Log the detection time and object name
                        log_file.write(f"{frame_count}, {current_time}, {object_name}\n")

                    # Draw bounding box and label
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert tensor to list of integers
                    confidence = box.conf.item()  # Convert tensor to scalar
                    cv2.rectangle(im0, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(im0, f"{object_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Apply object counting on the frame
            im0 = counter.start_counting(im0, results)

            # Write the processed frame to the output video
            video_writer.write(im0)

            # Display the frame with annotations (optional)
            cv2.imshow("Frame", im0)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")

# Release video capture and writer objects
cap.release()
video_writer.release()
cv2.destroyAllWindows()

# Write object counts to a file
output_counts_path = os.path.join(output_dir, "refrence07.txt")
with open(output_counts_path, "w") as file:
    for object_name, count in object_counts.items():
        file.write(f"{object_name}: {count}\n")

print(f"Object counts have been saved to {output_counts_path}")
print(f"Detection times have been saved to {log_path}")
print(f"Total frames processed: {frame_count}")
