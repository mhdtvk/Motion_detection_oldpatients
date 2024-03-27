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

def motion_detection(video_path : str):

    # Create a VideoCapture object to capture video from the camera or file
    cap = cv2.VideoCapture(video_path)

    # Initialize background model
    bg_model = None

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

        # Display original frame and motion mask
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Frame', thresh)
        cv2.imshow('Motion Mask', motion_frame)

        # Check for key press and break loop if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release VideoCapture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if int(input(" 0: Still \n 1: High Movment") ):
        motion_detection('/home/mt/Projects/internship/P3/motion detection/Negrar/0000/session_01/selection/CMVideo00076-high.avi')
    else:
        motion_detection('/home/mt/Projects/internship/P3/motion detection/Negrar/0000/session_01/selection/CMVideo00048-still.avi')
       