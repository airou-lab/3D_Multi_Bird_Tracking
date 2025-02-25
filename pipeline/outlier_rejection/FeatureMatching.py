import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import argparse
from KeypointExtractor import extract_keypoints

def load_keypoints(file_path):
    keypoints_array = np.load(file_path)
    keypoints = [cv.KeyPoint(x=kp[0], y=kp[1], _size=kp[2], _angle=kp[3], _response=kp[4], _octave=int(kp[5]), _class_id=int(kp[6])) for kp in keypoints_array]
    return keypoints

def load_descriptors(file_path):
    return np.load(file_path)

def load_images(img_path):
    return cv.imread(img_path)

def match_features(img1_path, img2_path, bbox_path_1, bbox_path_2, output_directory_path, show=True):
    image1 = load_images(img1_path)
    image2 = load_images(img2_path)

    print("Obtaining keypoints for first image")
    keypoints_1, descriptors_1 = extract_keypoints(img1_path, bbox_path_1, output_directory_path, show_keypoints=False, show_mask=False, save=False)
   
    print("Obtaining keypoints for second image")
    keypoints_2, descriptors_2 = extract_keypoints(img2_path, bbox_path_2, output_directory_path, show_keypoints=False, show_mask=False, save=False)


    print("Currently matching features between images ")
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv.drawMatches(image1, keypoints_1, image2, keypoints_2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if show:
        plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
        plt.title('Feature Matches')
        plt.show()

    return img3, keypoints_1, keypoints_2, matches



# Use like: python FeatureMatching.py frame_0615_1.jpg frame_0615_3.jpg cam1_frame615_boxes.csv cam3_frame615_boxes.csv

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Feature Matching')
#     parser.add_argument('img1_path', help='Path of the first image')
#     parser.add_argument('img2_path', help='Path of the second image')
#     parser.add_argument('bbox_name_1', help='Path of the detections for the camera corresponding to the first image')
#     parser.add_argument('bbox_name_2', help='Path of the detections for the camera corresponding to the second image')
#     args = parser.parse_args()

#     result, kp1, kp2, matches = match_features(args.img1_name, args.img2_name, args.bbox_name_1, args.bbox_name_2, show=True)


