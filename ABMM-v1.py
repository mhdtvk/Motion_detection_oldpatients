#ABMM
import cv2
import numpy as np

def optimize_motion_mask(fg_mask, min_thresh=0, kernel=np.array((9,9), dtype=np.uint8)):

    _, thresh = cv2.threshold(fg_mask,min_thresh,255,cv2.THRESH_BINARY)
    motion_mask = cv2.medianBlur(thresh, 3)

    # morphological operations
    kernel=np.array((9,9), dtype=np.uint8)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return motion_mask

# Function to calculate centroid
def calculate_centroid(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        
        if M['m10'] and M['m00'] and M['m01']:
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])
            return centroid_x, centroid_y , contours
    else:
        return None
    
def calculate_area(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = 0
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            area = cv2.contourArea(contour)
            areas += area
    return areas
    
def motion_per_pixels(motion_mask):
    # Count white pixels
    num_white_pixels = cv2.countNonZero(motion_mask)

    # Total number of pixels in the image
    total_pixels = motion_mask.shape[0] * motion_mask.shape[1]

    # Normalize the count
    normalized_count = num_white_pixels / total_pixels

    return normalized_count

def motion_detection(video_path : str):

    # Create a VideoCapture object to capture video from the camera or file
    cap = cv2.VideoCapture(video_path)

    # Initialize background model
    bg_model = None

    # Initialize variables for tracking movement
    previous_centroid = None
    total_movement = 0
    frame_count = 0
    displacement = 0
    total_movement_pix = 0


    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize background model with the first frame
        if bg_model is None:
            bg_model = gray.copy().astype("float")
            continue

        # Update background model using running average
        cv2.accumulateWeighted(gray, bg_model, 0.1)

        # Compute absolute difference between current frame and background model
        diff = cv2.absdiff(gray, cv2.convertScaleAbs(bg_model))

        # Apply thresholding to obtain binary motion mask
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        motion_frame = optimize_motion_mask(thresh)

        # Calculate centroid of motion mask
        centroid_info = calculate_centroid(motion_frame)
        area = calculate_area(motion_frame)
        print("\n========",area,"\n======")

        if centroid_info is not None:
            centroid_x, centroid_y, contours = centroid_info
            centroid = centroid_x, centroid_y

            frame = cv2.circle(frame, centroid, radius=5, color=(0, 0, 255), thickness=-1)  # Red marker

            if centroid and previous_centroid:
            # Calculate displacement between centroids
                displacement = ((centroid[0] - previous_centroid[0])**2 + (centroid[1] - previous_centroid[1])**2)**0.5
                total_movement += displacement

            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

            previous_centroid = centroid
        frame_count += 1

        motion_pixels = motion_per_pixels(motion_frame)
        total_movement_pix += motion_pixels
        average_movementp_per_frame = total_movement_pix / frame_count
        # Display original frame and motion mask
        average_movement_per_frame = total_movement / frame_count
        #print("\nThe displacement is :",frame_count ,average_movement_per_frame ,total_movement_pix , (average_movementp_per_frame*10**4))
        # Set the desired window sizes
        cv2.namedWindow('Original Frame', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Motion Mask', cv2.WINDOW_NORMAL)
        
        # Show the frames with the specified window sizes
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Frame', thresh)
        cv2.imshow('Motion Mask', motion_frame)
        width, height = 1000 , 800
        # Adjust the window sizes
        cv2.resizeWindow('Original Frame', width, height)
        cv2.resizeWindow('Frame', width, height)
        cv2.resizeWindow('Motion Mask', width, height)


        # Check for key press and break loop if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release VideoCapture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
    average_movement_per_frame = total_movement / frame_count
    # Print average movement per frame
    print("Average movement per frame:", average_movement_per_frame)



if __name__ == "__main__":
    if int(input(" 0: Still \n 1: High Movment") ):
        motion_detection('/home/mt/Projects/internship/P3/motion detection/Negrar/0000/session_01/selection/CMVideo00076-high.avi')
    else:
        motion_detection('/home/mt/Projects/internship/P3/motion detection/Negrar/0000/session_01/selection/CMVideo00048-still.avi')
       