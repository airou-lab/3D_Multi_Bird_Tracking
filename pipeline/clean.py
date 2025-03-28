import pandas as pd
import cv2 as cv
import numpy as np
import os
import KeypointExtractor as kp
import ContextBasedFeatureMatching as cbfm
import matplotlib.pyplot as plt
from collections import Counter
import random
import reconstruction as rec

camera_pairs = [(3, 5)]
camera_names = ['gopro1', 'gopro2', 'gopro3', 'gopro4', 'gopro5']
camera_frames_directories = [
    '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/video_frames/GoPro1_Encl4_03152024_1',
    '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/video_frames/GoPro2_Encl4_03152024_1',
    '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/video_frames/GoPro3_Encl4_03152024_1',
    '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/video_frames/GoPro4_Encl4_03152024_1',
    '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/video_frames/GoPro5_Encl4_03152024_1',
]

ssd_camera_frames_directories = [
    '/Volumes/T7/Encl4_03152024/GoPro1/GoPro1_Encl4_03152024_1/video_frames',
    '/Volumes/T7/Encl4_03152024/GoPro2/GoPro3_Encl4_03152024_1/video_frames',
    '/Volumes/T7/Encl4_03152024/GoPro3/GoPro3_Encl4_03152024_1/video_frames',
    '/Volumes/T7/Encl4_03152024/GoPro4/GoPro4_Encl4_03152024_1/video_frames',
    '/Volumes/T7/Encl4_03152024/GoPro5/GoPro5_Encl4_03152024_1/video_frames',
]

first_frames_paths = [
    '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/video_frames/GoPro1_Encl4_03152024_1/GoPro1_Encl4_03152024_1_frame_0000.jpg',
    '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/video_frames/GoPro2_Encl4_03152024_1/GoPro2_Encl4_03152024_1_frame_0000.jpg',
    '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/video_frames/GoPro3_Encl4_03152024_1/GoPro3_Encl4_03152024_1_frame_0000.jpg',
    '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/video_frames/GoPro4_Encl4_03152024_1/GoPro4_Encl4_03152024_1_frame_0000.jpg',
    '/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/video_frames/GoPro5_Encl4_03152024_1/GoPro5_Encl4_03152024_1_frame_0000.jpg',
]

trajectory_coords = []

class Feature():
    def __init__(self, feature, coordinate, bbox, matched, matched_feature, matched_bbox):
        self.feature = feature
        self.coordinate = coordinate
        self.bbox = bbox
        self.matched = matched
        self.matched_feature = matched_feature
        self.matched_bbox = matched_bbox

    def set_matched(self, matched):
        self.matched = matched

    def set_matched_feature(self, matched_feature):
        self.matched_feature = matched_feature



def rindex(lst, value):
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] == value:
            return i
    return -1

def locate_landmarks_in_images(image_paths, camera_num, save_landmarks=True):
    output_directory = f'{camera_frames_directories[camera_num-1]}/'
    landmarks = {
        "left-bottom1": [], "left-bottom2": [], "left-bottom3": [],
        "left-middle1": [], "left-middle2": [], "left-middle3": [],
        "left-top1": [], "left-top2": [], "left-top3": [],
        "right-bottom1": [], "right-bottom2": [], "right-bottom3": [],
        "right-middle1": [], "right-middle2": [], "right-middle3": [],
        "right-top1": [], "right-top2": [], "right-top3": [],
        "left-door1": [], "left-door2": [], "left-door3": [],
        "right-door1": [], "right-door2": [], "right-door3": [],
        "left-back1": [], "left-back2": [], "left-back3": [],
        "right-back1": [], "right-back2": [], "right-back3": [],
        "center-back1": [], "center-back2": [], "center-back3": [],
    }

    for image_path in image_paths:
        image = cv.imread(image_path)
        cv.imshow('Camera view', image)

        for landmark in landmarks:
            print(f'Current landmark: {landmark}')

            click_occurred = False

            def mouse_callback(event, x, y, flags, param):
                nonlocal click_occurred
                if event == cv.EVENT_LBUTTONDOWN:
                    print(f"Clicked coordinates for {landmark}: ({x}, {y})")
                    landmarks[landmark].append([x, y])
                    click_occurred = True

            cv.setMouseCallback('Camera view', mouse_callback)

            while not click_occurred:
                key = cv.waitKey(1) & 0xFF
                if key == ord('x'):
                    print(f'Skipped landmark {landmark}')
                    landmarks[landmark].append('none')
                    break

            click_occurred = False

        cv.destroyAllWindows()

    if save_landmarks:
        np.save(f'{output_directory}/selected_landmarks.npy', landmarks)
        print("Saved landmarks")

    return landmarks

def process_frame_bounding_boxes(image_frame_bounded_boxes, image_edges, image_mask, image_new_mask):
    for bbox_coords in image_frame_bounded_boxes:
        xmin, ymin, xmax, ymax, detection_count = int(bbox_coords[0]), int(bbox_coords[1]), int(bbox_coords[2]), int(bbox_coords[3]), bbox_coords[4]
        bbox_center_coords = ((xmin + xmax) / 2, (ymin + ymax) / 2)

        cv.rectangle(image_mask, (xmin, ymin), (xmax, ymax), 255, thickness=cv.FILLED)
        region_inside_edges = cv.bitwise_and(image_edges, image_edges, mask=image_mask)

        for y in range(region_inside_edges.shape[0]):
            row = region_inside_edges[y]
            start = -1
            end = -1

            for x in range(region_inside_edges.shape[1]):
                if row[x] != 0:
                    start = x
                    break

            if start >= 0:
                end = rindex(row, 255)

                if end > start:
                    for fill in range(start, end + 1):
                        image_new_mask[y, fill] = 255
                elif end == start:
                    image_new_mask[y, start] = 255

def map_keypoints_to_bounding_boxes(keypoints, bounding_boxes):
    keypoint_box_map = []
    bbox_dict_count = []
    for keypoint in keypoints:
        x, y = keypoint.pt
        for bbox in bounding_boxes:
            xmin, ymin, xmax, ymax, _ = bbox
            if xmin <= x <= xmax and ymin <= y <= ymax:
                keypoint_box_map.append((keypoint, bbox))
                bbox_dict_count.append(bbox)
                break

    return keypoint_box_map

def map_keypoints_to_bounding_boxes2(keypoints, bounding_boxes):
    keypoint_box_map = {keypoint: None for keypoint in keypoints}
    bbox_dict_count = []
    for keypoint in keypoints:
        x, y = keypoint.pt
        for bbox in bounding_boxes:
            xmin, ymin, xmax, ymax, _ = bbox
            if xmin <= x <= xmax and ymin <= y <= ymax:
                keypoint_box_map[keypoint] = (xmin, ymin, xmax, ymax)
                bbox_dict_count.append(bbox)
                break


    return keypoint_box_map

def main(camera1_number, camera2_number, start_frame=3, end_frame=7, select_new_landmarks=False, get_keypoints=True):
    image1_selected_landmarks = cbfm.select_landmarks(first_frames_paths[camera1_number-1], cbfm.landmark_names)
    image2_selected_landmarks = cbfm.select_landmarks(first_frames_paths[camera2_number-1], cbfm.landmark_names)

    detections1_path = f'/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/detections/clean_csvs/gopro{camera1_number}_clean.csv'
    detections2_path = f'/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/pipeline/detections/clean_csvs/gopro{camera2_number}_clean.csv'

    detections1 = pd.read_csv(detections1_path)
    detections2 = pd.read_csv(detections2_path)

    detections1['detection_count'] = detections1.groupby('frame_index').cumcount() + 1
    detections2['detection_count'] = detections2.groupby('frame_index').cumcount() + 1

    relevant_detections1 = detections1[(detections1["frame_index"] >= start_frame) & (detections1["frame_index"] <= end_frame)]
    relevant_detections2 = detections2[(detections2["frame_index"] >= start_frame) & (detections2["frame_index"] <= end_frame)]

    common_values1 = set(relevant_detections2["frame_index"].unique())
    common_values2 = set(relevant_detections1["frame_index"].unique())

    final_relevant_detections1 = relevant_detections1[relevant_detections1["frame_index"].isin(common_values1)]
    final_relevant_detections2 = relevant_detections2[relevant_detections2["frame_index"].isin(common_values2)]

    image1_frame_bounded_boxes = {num: [] for num in common_values1}
    image2_frame_bounded_boxes = {num: [] for num in common_values1}

    for index, detection in final_relevant_detections1.iterrows():
        xmin, ymin, xmax, ymax = detection['x1'], detection['y1'], detection['x2'], detection['y2']
        detection_count = detection['detection_count']
        image1_frame_bounded_boxes[detection['frame_index']].append((xmin, ymin, xmax, ymax, detection_count))

    for index, detection in final_relevant_detections2.iterrows():
        xmin, ymin, xmax, ymax = detection['x1'], detection['y1'], detection['x2'], detection['y2']
        detection_count = detection['detection_count']
        image2_frame_bounded_boxes[detection['frame_index']].append((xmin, ymin, xmax, ymax, detection_count))

    image1_paths = {}
    image2_paths = {}
    for frame_num in common_values1:
        frame_label = str(frame_num).zfill(4)

        ## Use for SSD paths
        image1_paths[frame_num] = f'{ssd_camera_frames_directories[camera1_number-1]}/_frame_{frame_label}.jpg'
        image2_paths[frame_num] = f'{ssd_camera_frames_directories[camera2_number-1]}/_frame_{frame_label}.jpg'
        
        ## Use for internal computer file paths
        # image1_paths[frame_num] = f'{camera_frames_directories[camera1_number-1]}/GoPro{camera1_number}_Encl4_03152024_1_frame_{frame_label}.jpg'
        # image2_paths[frame_num] = f'{camera_frames_directories[camera2_number-1]}/GoPro{camera2_number}_Encl4_03152024_1_frame_{frame_label}.jpg'

    total_kp_count = 0
    total_filtered_kp_count = 0

    kp_counts_img1 = []
    kp_counts_img2 = []

    kp_count_over_time = []
    rejections_over_time = []
    frame_counter = []


    save_imgs_with_keypoints = True

    large_mat = np.empty(0)

    for frame_index in common_values1:


        image1 = cv.imread(image1_paths[frame_index])
        image2 = cv.imread(image2_paths[frame_index])

        image1_gray = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
        image1_edges = cv.Canny(image1_gray, 100, 200)
        image2_gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
        image2_edges = cv.Canny(image2_gray, 100, 200)

        image1_mask = np.zeros_like(image1[:, :, 0])
        image1_new_mask = np.zeros_like(image1[:, :, 0])
        image2_mask = np.zeros_like(image2[:, :, 0])
        image2_new_mask = np.zeros_like(image2[:, :, 0])

        print("Obtaining masks for first camera frames")
        process_frame_bounding_boxes(image1_frame_bounded_boxes[frame_index], image1_edges, image1_mask, image1_new_mask)

        print("Obtaining masks for second camera frames")
        process_frame_bounding_boxes(image2_frame_bounded_boxes[frame_index], image2_edges, image2_mask, image2_new_mask)

        sift = cv.SIFT_create(nfeatures=5000)

        if get_keypoints:
            print("Extracting keypoints and descriptors...")
            image1_keypoints, image1_descriptors = sift.detectAndCompute(image1_new_mask, None)
            img1_with_keypoints = cv.drawKeypoints(image1, image1_keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            image2_keypoints, image2_descriptors = sift.detectAndCompute(image2_new_mask, None)
            img2_with_keypoints = cv.drawKeypoints(image2, image2_keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # plt.imshow(cv.cvtColor(img2_with_keypoints, cv.COLOR_BGR2RGB))
            # plt.show()

            image1_features = []
            image2_features = []

            if len(image1_keypoints) != 0:
                kp_counts_img1.append(len(image1_keypoints))

            if len(image2_keypoints) != 0:
                kp_counts_img2.append(len(image2_keypoints))
            
            print("Mapping keypoints to bounding boxes for first camera...")
            # keypoint_box_map1, majority_value_cam1, majority_occurence_cam1 = map_keypoints_to_bounding_boxes(image1_keypoints, image1_frame_bounded_boxes[frame_index])
            keypoint_box_map1 = map_keypoints_to_bounding_boxes(image1_keypoints, image1_frame_bounded_boxes[frame_index])
            for keypoint, bbox in keypoint_box_map1:
                print(f"Keypoint ({keypoint.pt[0]}, {keypoint.pt[1]}) in frame {frame_index} corresponds to bounding box {bbox}")
                image1_feature = Feature(keypoint, (keypoint.pt[0], keypoint.pt[1]), bbox, False, None, None)
                image1_features.append(image1_feature)

            print("Mapping keypoints to bounding boxes for second camera...")
            # keypoint_box_map2, majority_value_cam2, majority_occurence_cam2 = map_keypoints_to_bounding_boxes(image2_keypoints, image2_frame_bounded_boxes[frame_index])
            keypoint_box_map2 = map_keypoints_to_bounding_boxes(image2_keypoints, image2_frame_bounded_boxes[frame_index])
            for keypoint, bbox in keypoint_box_map2:
                print(f"Keypoint ({keypoint.pt[0]}, {keypoint.pt[1]}) in frame {frame_index} corresponds to bounding box {bbox}")
                image2_feature = Feature(keypoint, (keypoint.pt[0], keypoint.pt[1]), bbox, False, None, None)
                image2_features.append(image2_feature)

            keypoints_bbox_map1 = map_keypoints_to_bounding_boxes2(image1_keypoints, image1_frame_bounded_boxes[frame_index])
            keypoints_bbox_map2 = map_keypoints_to_bounding_boxes2(image2_keypoints, image2_frame_bounded_boxes[frame_index])


            # Feature matching between keypoints
            print("Matching keypoints...")
            bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
            matches = bf.match(image1_descriptors, image2_descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            print(f"Number of matches for frame {frame_index}: {len(matches)}")

            matched_keypoints_image1 = [image1_keypoints[m.queryIdx] for m in matches]
            matched_keypoints_image2 = [image2_keypoints[m.trainIdx] for m in matches]
            matched_keypoints_image1_coordinates = cv.KeyPoint_convert(matched_keypoints_image1)
            matched_keypoints_image2_coordinates = cv.KeyPoint_convert(matched_keypoints_image2)

            # Only triangulate 3D coordinates for images with a nonzero number of keypoints
            if len(matched_keypoints_image1) != 0:
                coords = rec.triangulate_points(matched_keypoints_image1_coordinates, matched_keypoints_image2_coordinates, rec.proj_matrix1, rec.proj_matrix2)
                trajectory_coords.append(coords)


            new_mat = []
            bbox_dict = {}
            for i in range(len(matched_keypoints_image1)):
                image1_keypoint = matched_keypoints_image1[i]
                image2_keypoint = matched_keypoints_image2[i]
                image1_keypoint_bbox = keypoints_bbox_map1[image1_keypoint]
                image2_keypoint_bbox = keypoints_bbox_map2[image2_keypoint]
                if image2_keypoint_bbox in bbox_dict.keys():
                    # new_mat.append(image2_keypoint_bbox)
                    bbox_dict[image2_keypoint_bbox] += 1
                else:
                    # new_mat.append(image2_keypoint_bbox)
                    bbox_dict[image2_keypoint_bbox] = 1

    # print(trajectory_coords)
    # rec.plot3d(trajectory_coords)


if __name__ == "__main__":
    for camera1, camera2 in camera_pairs:
        main(camera1, camera2, start_frame=1, end_frame=5, select_new_landmarks=False, get_keypoints=True)
    rec.plot3d(trajectory_coords)

