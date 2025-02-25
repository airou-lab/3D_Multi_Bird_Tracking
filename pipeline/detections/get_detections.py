import pandas as pd
import subprocess
import os


def run_detection_script(video_file_path):
    # Save the current working directory
    original_dir = os.getcwd()
    
    try:
        # Change directory to the desired path
        os.chdir(os.path.join('..', 'yolo', 'yolov5'))
        
        # Define the command and its arguments
        command = [
            'python', 'detect.py',
            '--weights', 'runs/train/exp7/weights/best.pt',
            '--source', video_file_path,
            '--conf', '0.25',
            '--save-csv', True,

        ]
        
        # Run the command
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Check the result
        if result.returncode == 0:
            print("Script executed successfully!")
            print("Output:", result.stdout)
        else:
            print("Script failed with error:")
            print("Error:", result.stderr)
    
    finally:
        # Return to the original directory
        os.chdir(original_dir)








# video_file_paths = [
#             # '/Volumes/T7/Encl4_03152024/GoPro1/GoPro1_Encl4_03152024_1.MP4',
#             # '/Volumes/T7/Encl4_03152024/GoPro2/GoPro2_Encl4_03152024_1.MP4',
#             # '/Volumes/T7/Encl4_03152024/GoPro3/GoPro3_Encl4_03152024_1.MP4',
#             # '/Volumes/T7/Encl4_03152024/GoPro4/GoPro4_Encl4_03152024_1.MP4',
#             # '/Volumes/T7/Encl4_03152024/GoPro5/GoPro5_Encl4_03152024_1.MP4',
# ]


# for video in video_file_paths:
#     run_detection_script(video)



def clean_csv(file_path):
    # Load the CSV file without headers
    df = pd.read_csv(file_path, header=None)
    print("Initial data load:")
    print(df.head())  # Inspect the first few rows to understand the structure

    # Specifying column names based on your observed data structure
    column_names = ['Frame', 'Path', 'Extra', 'Label', 'Confidence', 'xmin', 'ymin', 'xmax', 'ymax']
    df.columns = column_names[:len(df.columns)]  # Adjust column names to match data

    # Drop any rows where critical data might be missing (e.g., no bounding box coordinates)
    df.dropna(subset=['xmin', 'ymin', 'xmax', 'ymax'], inplace=True)

    # Correct data shifting issues by checking data types or misplacements
    if df['xmin'].dtype == object:
        # Assume 'xmin' should be float, and non-float entries indicate row misalignment
        df = df[df['xmin'].apply(lambda x: x.replace('.', '', 1).isdigit())]

    df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce')
    df[['xmin', 'ymin', 'xmax', 'ymax']] = df[['xmin', 'ymin', 'xmax', 'ymax']].apply(pd.to_numeric, errors='coerce')

    print("Cleaned DataFrame head:")
    print(df.head())

    # Save the cleaned data to a new CSV file
    df.to_csv(file_path, index=False)




# clean_csv('/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/detections/GoPro5_Encl4_03152024_1.MP4/detections.csv')



# # Usage: (path to csv you want to be organized, new path to the new csv name and where you want it to be placed)
# clean_csv('/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/yolov5/runs/detect/Angle3_Output/predictions.csv', 
#           '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/csv/cleaned_output3.csv')