import cv2
import os
import numpy

# Saves video frames and returns directory of saved frames for corresponding video input

# Can be easily run on videos on external SSD
def video_to_frames(video_file_path, max_frames=100, desired_fps=30):
    
    output_directory_name = video_file_path[-27:-4]
    # Create a directory to save the frames
    os.makedirs(output_directory_name, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_file_path)

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set the desired frame rate (30 FPS)
    desired_fps = 30

    # Calculate the frame interval for capturing frames
    frame_interval = max(1, round(fps / desired_fps))

    # Initialize variables
    frame_count = 0
    extracted_frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break  
        # Break the loop when we reach the end of the video

        # Save the frame as an image in the output directory at the desired frame rate
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_directory_name, f'{output_directory_name}_frame_{extracted_frame_count:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            extracted_frame_count += 1

        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f'{extracted_frame_count} frames extracted at 30 frames per second and saved to {output_directory_name}')
    
    return output_directory_name


video_file_paths = [
            '/Volumes/T7/Encl4_03152024/GoPro1/GoPro1_Encl4_03152024_1.MP4',
            # '/Volumes/T7/Encl4_03152024/GoPro2/GoPro2_Encl4_03152024_1.MP4',
            # '/Volumes/T7/Encl4_03152024/GoPro3/GoPro3_Encl4_03152024_1.MP4',
            # '/Volumes/T7/Encl4_03152024/GoPro4/GoPro4_Encl4_03152024_1.MP4',
            # '/Volumes/T7/Encl4_03152024/GoPro5/GoPro5_Encl4_03152024_1.MP4',
]


for video in video_file_paths:
    video_to_frames(video)





