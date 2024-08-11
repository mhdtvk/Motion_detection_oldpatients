from json_report_md import *
from moviepy.editor import VideoFileClip

import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

class ShowFrames :
    def __init__(self, md_trsh, xlim, ylim) -> None:
        self.md_trsh = md_trsh
        # Create a new figure with subplots
        self.fig, self.axes = plt.subplots(figsize=(14, 4))
        self.threshold_line = self.axes.axhline(y=self.md_trsh, color='blue', linestyle='--', label='Threshold')
        self.plot_line, = self.axes.plot([], [],lw=1, color='red')
        self.axes.set_title('Frame/Motion amount')
        self.axes.set_xlim(0, xlim)  
        self.axes.set_ylim(0, ylim)  
        plt.tight_layout()
        # Define the font, position, and scale of the text on the frames
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.position1 = (50, 50)  # Position of the text (x, y)
        self.position2 = (50, 100)  # Position of the text (x, y)
        self.font_scale = 1  # Font scale (size)
        self.font_color = (255, 255, 255)  # Font color in BGR format
        self.thickness = 2  # Thickness of the text

    def show(self, orgframe, motionmask, videoname = 'no_name', frame_number = 0, areas = 0, video_time = 0.0):
        text = f'Video name: {videoname} Frame: {frame_number} Motion amount: {areas} Time(s) : {video_time}'
        cv2.putText(orgframe, text, self.position1, self.font, self.font_scale, self.font_color, self.thickness)
        text_mood = f'Limit: {self.md_trsh} Motion Detected : {"Yes" if areas >= self.md_trsh else "No"}'
        cv2.putText(orgframe, text_mood, self.position2, self.font, self.font_scale, (0, 0, 255) if areas >= self.md_trsh else (255, 255, 255), self.thickness)
        self.update_mf_plot ((frame_number, areas))
        cv2.namedWindow(f'Original Frame {videoname}', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Motion Mask', cv2.WINDOW_NORMAL)
        # Show the frames with the specified window sizes
        cv2.imshow(f'Original Frame {videoname}', orgframe)
        cv2.imshow('Motion Mask', motionmask)
        # Set the desired window sizes
        width, height = 900 , 500
        # Adjust the window sizes
        cv2.resizeWindow(f'Original Frame {videoname}', width, height)
        cv2.resizeWindow('Motion Mask', width, height)
        # Check for key press and break loop if 'q' is pressed
        key = cv2.waitKey(30)
        if key & 0xFF == ord('q'):
            exit(0)
        
    def update_mf_plot(self,new_data_point ):
        x_data, y_data = self.plot_line.get_data()
        x_data = list(x_data) + [new_data_point[0]]
        y_data = list(y_data) + [new_data_point[1]]
        self.plot_line.set_data(x_data, y_data)
        self.axes.relim()  # Recalculate limits
        self.axes.autoscale_view()  # Autoscale
        plt.draw()
        plt.pause(0.00005)


class MotionDetectionABMM :
    def __init__(self) :
        self.framenum_dataset = None
        self.motions_amount_dataset = None
        self.md_trsh = None
        self.json_report = JSONReportMD() 
        self.motion_detected = False

    def optimize_motion_mask(self, fg_mask, min_thresh=0):

        _, thresh = cv2.threshold(fg_mask,min_thresh,255,cv2.THRESH_BINARY)
        motion_mask = cv2.medianBlur(thresh, 3)
        motion_mask = cv2.medianBlur(motion_mask, 3)
        # morphological operations
        kernel=np.array((9,9), dtype=np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return motion_mask
    
    # Function to calculate centroid
    def calculate_contours(self, binary_mask):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        areas = 0
        if contours:
            for contour in contours:
                M = cv2.moments(contour)
                if M['m00']:
                    area = cv2.contourArea(contour)
                    areas += area
            return areas, contours
        else:
            return 0, []

    def motion_detection(self, video_path : str, lr = 0.09, trsh = 25, md_trsh = 2500, ylim = 10000, show = True, json_gen = True):
        self.md_trsh = md_trsh
        video = VideoFileClip(video_path)
        frame_rate = video.fps
        tot_num_frm = int(video.duration * frame_rate)
        video_time = 0.0
        video_name = os.path.basename(video_path)
        # Initialize variables for tracking movement
        areas = 0
        frame_number = -1
        motion_frames_count = 0
        if show :
            showmotions = ShowFrames(md_trsh, tot_num_frm, ylim)
        # Initialize background model
        bg_model = None
        for frame_idx, frame_org in enumerate(video.iter_frames()) :           
            frame_number += 1
            frame = frame_org.copy()
            video_time = frame_number / frame_rate
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Initialize background model with the first frame
            if bg_model is None:
                bg_model = gray.copy().astype("float")
                continue
            # Update background model using running average
            cv2.accumulateWeighted(gray, bg_model, lr)
            # Compute absolute difference between current frame and background model
            diff = cv2.absdiff(gray, cv2.convertScaleAbs(bg_model))
            # Apply thresholding to obtain binary motion mask
            _, thresh = cv2.threshold(diff, trsh, 255, cv2.THRESH_BINARY)
            motion_frame = self.optimize_motion_mask(thresh,min_thresh=0)
            # Calculate centroid of motion mask
            areas, contours = self.calculate_contours(motion_frame)

            if contours :
                cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

            self.motion_detected =  areas > self.md_trsh 
            # Sending frame data to the JSON logger Class
            if json_gen:
                self.json_report.frame_info_gen(frame_num= frame_number, time_stmp= video_time, motion_amount= areas, motion_detected= True)
            
            if self.motion_detected:
                # Counting the number of frames that has movment
                motion_frames_count += 1
          
            if show:
                showmotions.show(frame, motion_frame, video_name, frame_number, areas, video_time)

        # Sending the video data to Json Logger
        if json_gen:
            self.json_report.alghorithm_info_gen(alg_name= "ABMM", lr= lr, md_trsh= md_trsh)
            self.json_report.video_info_gen(video_path = video_path, frames_with_motion = motion_frames_count)
            self.json_report.save_json_file()
            self.json_report.plot_json()
        # close all windows
        cv2.destroyAllWindows() 


if __name__ == "__main__":
    
    folder_path = input("The path of the video files: ")
    lr = float(input("\n Learning Rate : "))
    trsh = int(input("\n Threshold : "))
    md_trsh = int(input("\n Threshold for motion detection: "))

    choice = int(input("\n".join([
    "\n Choose the Operation: ",
    " 1- Json Report Generator and Plotting : Motion per Frame",
    " 2- Showing Frames : Motion Detected frames and Plot\n",
    ])))

    if choice == 1:        
        file_list = glob.glob(folder_path + '/*')  
        for file_path in file_list :
            motion_detection = MotionDetectionABMM()
            print("New File Generating...")
            motion_detection.motion_detection(file_path, lr, trsh, md_trsh=md_trsh, show = False, json_gen= True)
            
    elif choice == 2:
        ylim = int(input("\n The high range for displaying detected movements in plotting: "))
        
        file_list = glob.glob(folder_path + '/*')  
        for file_path in file_list :
            motion_detection = MotionDetectionABMM()
            print("New File Generating...")
            motion_detection.motion_detection(file_path, lr, trsh, md_trsh=md_trsh, ylim= ylim, show = True, json_gen= False)
