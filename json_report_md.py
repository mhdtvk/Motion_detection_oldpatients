from json_plotter import *
import cv2
import os
import json

class JSONReportMD:
    def __init__(self) -> None:
        self.video_info = {'name' : '',
                           'resolution' : '',
                           'fps' : '',
                           'duration_(sec)' : '',
                           'total_frames' : '',
                           'frames_with_motion' : '',
                           'Movement_percentage %' : ''
                           }
        
        self.algorithms = []
        self.frames = []
        self.json_name = None
        self.file_path = None

    def video_info_gen(self, video_path, num_frames_with_motion):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return None

        self.video_info['name'] = os.path.basename(video_path)
        self.json_name = os.path.basename(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.video_info['resolution'] = (width, height)
        self.video_info['fps'] = int(cap.get(cv2.CAP_PROP_FPS))
        self.video_info['total_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_info['duration_(sec)'] = self.video_info['total_frames'] / self.video_info['fps']
        self.video_info['frames_with_motion'] = num_frames_with_motion
        self.video_info['Movement_percentage %'] = round(((self.video_info['frames_with_motion'] / self.video_info['total_frames']) * 100),2)

        cap.release()

    def alghorithm_info_gen(self, alg_name, lr, md_trsh) :
        new_alghorithm_stat = True

        for alghorithms in self.algorithms:
            if alghorithms['name'] == alg_name and alghorithms['LearningRate'] == lr and alghorithms['motion_detection_threshold'] == md_trsh :
                new_alghorithm_stat = False

        if new_alghorithm_stat :
            new_alghorithm = {
                            'id' : len(self.algorithms),
                            'name' : alg_name,
                            'LearningRate' : lr,
                            'motion_detection_threshold' : md_trsh
                            }
            self.algorithms.append(new_alghorithm)

    def frame_info_gen(self, frame_num : int, time_stmp : str, motion_amount : int, motion_detected : bool)  :
        new_frame = {
                    'frame_number' : frame_num,
                    'time_stamp_(sec)' : time_stmp,
                    'motion_amount' : motion_amount,
                    'motion_detected' : motion_detected
                    }
        
        self.frames.append(new_frame)
        
    def save_json_file(self, save_path= ''):
        report = {
                'video_info' : self.video_info,
                'algorithms' : self.algorithms,
                'frames' : self.frames
                }
        # Directory where you want to save the JSON file
        directory = "Json_report"
        
        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # File path where you want to save the JSON file
        self.file_path = f"{directory}/{self.json_name}.json"

        # Write the list of dictionaries to the JSON file
        with open(self.file_path, "w") as json_file:
            json.dump(report, json_file, indent=4)

        print("JSON file saved successfully.","\nPath:self.file_path")

    def plot_json(self) :
        json_plotter = JsonPlotter(self.file_path, self.json_name)
        json_plotter.run()
        
        
    


        



        

        
