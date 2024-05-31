from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import os

# To avoid any potential issues with duplicate libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Initialize video capture (0 for the first webcam connected)
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"

# Get video properties
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points to cover the entire frame
region_points = [(0, 0), (w, 0), (w, h), (0, h)]

# Initialize video writer
video_writer = cv2.VideoWriter(
    "count.avi",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (w, h)
)

# Initialize Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(
    view_img=True,
    reg_pts=region_points,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=2
)

# Initialize dictionary to count objects
object_counts = {}

# Start processing the video frames
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Perform object tracking on the frame
    results = model.track(im0, persist=True, show=False)

    # Count objects in the current frame
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls)  # Get the class ID
            object_name = model.names[class_id]

            if object_name not in object_counts:
                object_counts[object_name] = 0  # Initialize if not present

            object_counts[object_name] += 1

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
with open("Aobject_counts.txt", "w") as file:
    for object_name, count in object_counts.items():
        file.write(f"{object_name}: {count}\n")

print("Object counts have been saved to Aobject_counts.txt")
