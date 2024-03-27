#MOG2
import cv2
import numpy as np



def optimize_motion_mask(fg_mask, min_thresh=0, kernel=np.array((9,9), dtype=np.uint8)):

    _, thresh = cv2.threshold(fg_mask,25,255,cv2.THRESH_BINARY)
    motion_mask = cv2.medianBlur(thresh, 3)
    motion_mask = cv2.medianBlur(motion_mask, 3)
    # morphological operations
    kernel=np.array((9,9), dtype=np.uint8)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return motion_mask

    
def motion_detection(video_path : str):
    # Create a VideoCapture object to capture video from the camera or file
    cap = cv2.VideoCapture(video_path)

    # Create a BackgroundSubtractorMOG2 object for background subtraction
    fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Apply background subtraction to detect motion
        fgmask = fgbg.apply(frame)
        opt_fgmask = optimize_motion_mask(fgmask)

        # Display the original frame and the motion masks
        cv2.imshow('Original Frame', frame)
        cv2.imshow('FG Mask', fgmask)
        cv2.imshow('Optimized Motion Mask', opt_fgmask)

        # Check for key press and break loop if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close all windows
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    if int(input(" 0: Still \n 1: High Movment") ):
        motion_detection('/home/mt/Projects/internship/P3/motion detection/Negrar/0000/session_01/selection/CMVideo00076-high.avi')
    else:
        motion_detection('/home/mt/Projects/internship/P3/motion detection/Negrar/0000/session_01/selection/CMVideo00048-still.avi')