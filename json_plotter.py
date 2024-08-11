import json
import matplotlib.pyplot as plt
import os

class JsonPlotter :
    def __init__(self, file_path, file_name) :
        self.json_path = file_path
        self.file_name = file_name
        self.json_file = None
        self.frames_number = []
        self.motion_amounts = []
        self.timestamps = []

    def load_json(self) :
        with open(self.json_path, "r") as json_files:
            self.json_file = json.load(json_files)

    def data_extractor(self) :
        frames = self.json_file['frames']
        for frame in frames :
            self.frames_number.append(frame['frame_number'])
            self.motion_amounts.append(frame['motion_amount'])
            self.timestamps.append(frame['time_stamp_(sec)'])

    def data_plotter(self):
        fig = plt.figure(figsize=(30, 12))
        plt.plot(self.frames_number, self.motion_amounts, color='blue')
        plt.axhline(y=2500, color='red', linestyle='--', label='Motion Detection Threshold')
        plt.xlabel('Frames')
        plt.ylabel('Motion Amount')
        plt.title('Motion Amount per Frame')
        plt.legend()
        
        # Create the directory if it doesn't exist
        plot_dir = os.path.join(os.path.dirname(self.json_path), 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Save the plot in the specified directory
        plot_path = os.path.join(plot_dir, f'{self.file_name}.png')
        plt.savefig(plot_path)  
        plt.close()
        print("Plot Saved.")


    def run(self) :
        self.load_json()
        self.data_extractor()
        self.data_plotter()


