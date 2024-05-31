#problem of repetetive counting solved here. Count stored in txt file
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import os

# Avoid potential issues with duplicate libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Set the confidence threshold to 90%
model.conf = 0.9

# Initialize video capture (0 for the first webcam connected or provide video path)
video_path = "HD02.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception(f"Error reading video file {video_path}")

# Get video properties
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points to cover the entire frame
region_points = [(0, 0), (w, 0), (w, h), (0, h)]

# Initialize video writer
output_path = "optxt/refrence3.avi"
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

# Start processing the video frames
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    frame_count += 1  # Increment frame count

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

# Release video capture and writer objects
cap.release()
video_writer.release()
cv2.destroyAllWindows()

# Write object counts to a file
output_counts_path = "optxt/refrennce3.txt"
with open(output_counts_path, "w") as file:
    for object_name, count in object_counts.items():
        file.write(f"{object_name}: {count}\n")

print(f"Object counts have been saved to {output_counts_path}")
print(f"Total frames processed: {frame_count}")
