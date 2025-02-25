import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.path import Path
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
from BirdsNearestLandmarks import select_landmarks, landmark_names
from FeatureMatching import match_features


# Load keypoints from .npy file, give file path
def load_keypoints(file_path):
    keypoints_array = np.load(file_path, allow_pickle=True)
    keypoints = [cv.KeyPoint(x=kp[0], y=kp[1], size=kp[2], angle=kp[3], response=kp[4], octave=int(kp[5]), class_id=int(kp[6])) for kp in keypoints_array]
    return keypoints


def get_keypoint_coordinates(file_path):
    keypoints = load_keypoints(file_path=file_path)
    keypoint_coords = cv.KeyPoint_convert(keypoints)
    print(f'Obtained coordinates for {len(keypoint_coords)} keypoints.')
    return tuple(keypoint_coords)


def match_keypoints_to_closest_landmark(keypoint_coords, selected_landmarks, image_path, show=False):
    num_coords = len(keypoint_coords)
    keypoint_landmark_dictionary = {
        num: None for num in range(num_coords)
    }
    closest_landmarks = []
    closest_distances = []
    keypoint_num = 0
    image = cv.imread(image_path)

    for keypoint_coord in keypoint_coords:
        bird_center = np.array(keypoint_coord)
        x = int(keypoint_coord[0])
        y = int(keypoint_coord[1])
        coord = (x, y)

        min_dist = float('inf')
        closest_landmark = None
        
        for name, landmark_coord in selected_landmarks.items():
            if landmark_coord is None or len(landmark_coord) == 0 or landmark_coord == 'none':
                continue  # Skip if landmark not selected
            dist = np.linalg.norm(bird_center - np.array(landmark_coord))
            if dist < min_dist:
                min_dist = dist
                closest_landmark = name
        
        closest_landmarks.append(closest_landmark)
        closest_distances.append(min_dist)

        cv.line(image, coord, selected_landmarks[closest_landmark], (0, 255, 0), 2)
        cv.circle(image, coord, 5, (0, 0, 255), -1)
        cv.circle(image, selected_landmarks[closest_landmark], 5, (255, 0, 0), -1)
        cv.putText(image, f'{closest_landmark}', selected_landmarks[closest_landmark], cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add this line back in to display distances from each keypoint to its nearest landmark
        if keypoint_num % 5 == 0:
            cv.putText(image, f'Keypoint {keypoint_num}: {coord}', coord, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        keypoint_landmark_dictionary[keypoint_num] = [closest_landmark, coord]

        keypoint_num += 1
    
    if show:
        cv.imshow("Birds and Landmarks", image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # cv.imwrite(image, f'landmarks_{image_path[7:-4]}.jpg')

    return closest_landmarks, closest_distances, keypoint_landmark_dictionary


def get_keypoint_match_pairs(kp1, kp2, matches, image1_path, image2_path, selected_landmarks1, selected_landmarks2, show=False):
    image1 = cv.imread(image1_path)
    image2 = cv.imread(image2_path)
    
    print("Number of initial keypoints in image1")
    print(len(kp1))
    print("Number of initial keypoints in image2")
    print(len(kp2))
    
    matched_keypoints1 = [kp1[m.queryIdx] for m in matches]
    matched_keypoints2 = [kp2[m.trainIdx] for m in matches]

    print("Coordinates of matched keypoints")
    print(matched_keypoints1)
    print(matched_keypoints2)
    
    matched_keypoints1_coordinates = cv.KeyPoint_convert(matched_keypoints1)
    matched_keypoints2_coordinates = cv.KeyPoint_convert(matched_keypoints2)

    print("Number of matched keypoints")
    print(len(matched_keypoints1_coordinates))

    _, __, keypoints1_landmark_dictionary = match_keypoints_to_closest_landmark(matched_keypoints1_coordinates, selected_landmarks=selected_landmarks1, image_path=image1_path)
    _, __, keypoints2_landmark_dictionary = match_keypoints_to_closest_landmark(matched_keypoints2_coordinates, selected_landmarks=selected_landmarks2, image_path=image2_path)


    final_keypoints1 = []
    final_keypoints2 = []
    final_matches = []
    matches_to_keep_indices = []

    for landmark_num in range(len(matched_keypoints1_coordinates)):
        if keypoints1_landmark_dictionary[landmark_num][0] == keypoints2_landmark_dictionary[landmark_num][0]:
            matches_to_keep_indices.append(landmark_num)
            # continue

        else:
            keypoints1_landmark_dictionary[landmark_num] = None
            keypoints2_landmark_dictionary[landmark_num] = None

    for index, matched_kp1s in enumerate(matched_keypoints1):
        if keypoints1_landmark_dictionary[index] is not None:
            final_keypoints1.append(matched_kp1s)
            final_matches.append(matches[index]) 

    for index, matched_kp1s in enumerate(matched_keypoints2):
        if keypoints2_landmark_dictionary[index] is not None:
            final_keypoints2.append(matched_kp1s) 

    print(f"Final keypoints 1 count: {len(final_keypoints1)}")
    print(f"Final keypoints 2 count: {len(final_keypoints2)}")  
    print(f"Final matches count: {len(final_matches)}")

    filtered_matches = [m for i, m in enumerate(matches) if i in matches_to_keep_indices]

    filtered_keypoints1 = [kp1[m.queryIdx] for m in filtered_matches]
    filtered_keypoints2 = [kp2[m.trainIdx] for m in filtered_matches]

    filtered_keypoints1_coords = cv.KeyPoint_convert(filtered_keypoints1)
    filtered_keypoints2_coords = cv.KeyPoint_convert(filtered_keypoints2)

    if show:
        img3 = cv.drawMatches(image1, kp1, image2, kp2, filtered_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
        plt.title('Feature Matches')
        plt.show()
        plt.imsave('/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/matched_features.png', img3)
        

    return filtered_keypoints1_coords, filtered_keypoints2_coords






# keypoints1_path = 'keypoints/frame_0615_1_keypoints.npy'
# keypoints2_path = 'keypoints/frame_0615_3_keypoints.npy'

# image1_path = 'images/frame_0615_1.jpg'
# image2_path = 'images/frame_0615_3.jpg'

# image1_name = 'frame_0615_1.jpg'
# image2_name = 'frame_0615_3.jpg'

# birds1_path = 'bounded_boxes/cam1_frame615_boxes.csv'
# birds2_path = 'bounded_boxes/cam3_frame615_boxes.csv'

# birds1_name = 'cam1_frame615_boxes.csv'
# birds2_name = 'cam3_frame615_boxes.csv'

# img, kp1, kp2, matches = match_features(img1_name=image1_name, img2_name=image2_name, bbox_name_1=birds1_name, bbox_name_2=birds2_name, show=False)
# selected_landmarks1 = select_landmarks(image_path=image1_path, landmark_names=landmark_names)
# selected_landmarks2 = select_landmarks(image_path=image2_path, landmark_names=landmark_names)

# output1, output2 = get_keypoint_match_pairs(kp1=kp1, kp2=kp2, matches=matches, image1_path=image1_path, image2_path=image2_path, selected_landmarks1=selected_landmarks1, selected_landmarks2=selected_landmarks2 )

# python detect.py --weights runs/train/exp8/weights/best.pt --source /Volumes/T7/Encl4_03152024/GoPro1/GoPro1_Encl4_03152024_1.MP4 --conf 0.25 --view-img --save-csv

#  python detect.py --weights runs/train/exp8/weights/best.pt --source /Volumes/T7/Encl4_03152024/GoPro2/GoPro2_Encl4_03152024_1.MP4 --conf 0.25 --view-img --save-csv

#  python detect.py --weights runs/train/exp8/weights/best.pt --source /Volumes/T7/Encl4_03152024/GoPro3/GoPro3_Encl4_03152024_1.MP4 --conf 0.25 --view-img --save-csv

#  python detect.py --weights runs/train/exp8/weights/best.pt --source /Volumes/T7/Encl4_03152024/GoPro4/GoPro4_Encl4_03152024_1.MP4 --conf 0.25 --view-img --save-csv

#  python detect.py --weights runs/train/exp8/weights/best.pt --source /Volumes/T7/Encl4_03152024/GoPro5/GoPro5_Encl4_03152024_1.MP4 --conf 0.25 --view-img --save-csv


