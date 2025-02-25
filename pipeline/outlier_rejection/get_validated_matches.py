import cv2 as cv
import argparse
import ContextBasedFeatureMatching as cb
import FeatureMatching as fm

def main(img1_path, img2_path, detections1_name, detections2_name, show=False):

    output_directory_path = '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/outlier_rejection'


    print("Currently matching image features")
    result, kp1, kp2, matches = fm.match_features(img1_path, img2_path, detections1_name, detections2_name, output_directory_path=output_directory_path, show=show)


    print("Currently selecting local landmarks")
    selected_landmarks1 = cb.select_landmarks(img1_path, cb.landmark_names)
    selected_landmarks2 = cb.select_landmarks(img2_path, cb.landmark_names)


    print("Currently applying outlier rejection")
    filtered_keypoints1, filtered_keypoints2 = cb.get_keypoint_match_pairs(kp1=kp1, kp2=kp2, matches=matches, img1_path=img1_path, img2_path=img2_path, selected_landmarks1=selected_landmarks1, selected_landmarks2=selected_landmarks2)

    filtered_keypoints1_coords = tuple(cv.KeyPoint_convert(filtered_keypoints1))
    filtered_keypoints2_coords = tuple(cv.KeyPoint_convert(filtered_keypoints2))
    print("Finished, got remaining valid keypoint coordinates")


# Use like: python get_validated_matches.py '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/video_frames/GoPro1_Encl4_03152024_1/GoPro1_Encl4_03152024_1_frame_0000.jpg' '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/video_frames/GoPro3_Encl4_03152024_1/GoPro3_Encl4_03152024_1_frame_0000.jpg' '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/detections/GoPro1_Encl4_03152024_1.MP4/detections.csv' '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/detections/GoPro3_Encl4_03152024_1.MP4/detections.csv' 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img1_path", help="Enter path of first image")
    parser.add_argument("img2_path", help="Enter path of second image")
    parser.add_argument("camera1_detections_path", help="Enter path of detections for first camera view")
    parser.add_argument("camera2_detections_path", help="Enter path of detections for second camera view")
    # parser.add_argument("show", help="Boolean default to False, if True shows validated feature matches")

    args = parser.parse_args()

    main(args.img1_path, args.img2_path, args.camera1_detections_path, args.camera2_detections_path, show=True)