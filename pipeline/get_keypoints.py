import os
import KeypointExtractor as kp
import argparse


def get_keypoints(GoPro_frames_folder, GoPro_detections_path):
    # detections_path = kp.load_bounded_boxes(GoPro_detections_path)

    # Set output directory path (set to current keypoints folder)
    output_directory_path = '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/video_frames/video_frames_with_keypoints'

    # Get the working directory
    folder_path = GoPro_frames_folder

    # List all images in the GoPro_frames_folder directory
    for filename in os.listdir(folder_path):
        # Create full file path
        file_path = os.path.join(folder_path, filename)
        print(f'{file_path} image loaded')
        # Check to make sure it is an image
        if os.path.isfile(file_path) and filename.lower().endswith(('.jpg')):
            # Gets and saves keypoints and descriptors, and saves image with keypoints for each frame
            # keypoints, descriptors = kp.extract_keypoints(file_path, GoPro_detections_path, output show_keypoints=False, show_mask=False, save=True)
            keypoints, descriptors = kp.extract_keypoints(file_path, filename, GoPro_detections_path, output_directory_path=output_directory_path, show_keypoints=False, show_mask=False, save=True, save_img_with_keypoints=False)
            print(f'keypoints extracted for {file_path}')


# def main(GoPro_frames_folder_path, GoPro_detections_path):
#     print('in main')
#     get_keypoints(args.GoPro_frames_folder_path, args.GoPro_detections_path)
            

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("GoPro_frames_folder_path", help="Enter GoPro_frames_folder path")
#     parser.add_argument("GoPro_detections_path", help="Enter GoPro_detections_folder path")

#     args = parser.parse_args()


#     main(GoPro_frames_folder_path=args.GoPro_frames_folder_path, GoPro_detections_path=args.GoPro_detections_path)
#     # get_keypoints(args.GoPro_frames_folder_path, args.GoPro_detections_path)
    


# Use like: python get_keypoints.py GoPro_frames_folder_path GoPro_detections_path
    
# python get_keypoints.py '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/video_frames/GoPro1_Encl4_03152024_1' '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/detections/GoPro1_Encl4_03152024_1.MP4/detections.csv'
# python get_keypoints.py /Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/video_frames/GoPro1_Encl4_03152024_1 /Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/detections/GoPro1_Encl4_03152024_1.MP4/detections.csv

# def get_keypoints(detections_path): 

#     for image in 



#     bounded_boxes = kp.load_bounded_boxes(detections_path)





# detection_folders = [
#     '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/detections/GoPro1_Encl4_03152024_1.MP4/detections.csv',
#     '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/detections/GoPro2_Encl4_03152024_1.MP4/detections.csv',
#     '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/detections/GoPro3_Encl4_03152024_1.MP4/detections.csv',
#     '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/detections/GoPro4_Encl4_03152024_1.MP4/detections.csv',
#     '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/detections/GoPro5_Encl4_03152024_1.MP4/detections.csv',
# ]