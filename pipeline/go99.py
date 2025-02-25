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


    # majority_value, majority_count = Counter(bbox_dict_count).most_common(1)[0]

    # return keypoint_box_map, majority_value, majority_count
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


    # majority_value, majority_count = Counter(bbox_dict_count).most_common(1)[0]

    # return keypoint_box_map, majority_value, majority_count
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
        # image1_paths[frame_num] = f'{ssd_camera_frames_directories[camera1_number-1]}/_frame_{frame_label}.jpg'
        # image2_paths[frame_num] = f'{ssd_camera_frames_directories[camera2_number-1]}/_frame_{frame_label}.jpg'
        
        ## Use for internal computer file paths
        image1_paths[frame_num] = f'{camera_frames_directories[camera1_number-1]}/GoPro{camera1_number}_Encl4_03152024_1_frame_{frame_label}.jpg'
        image2_paths[frame_num] = f'{camera_frames_directories[camera2_number-1]}/GoPro{camera2_number}_Encl4_03152024_1_frame_{frame_label}.jpg'

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


            # majority_bbox2, majority_count2 = find_majority_bbox(keypoint_box_map2)
            # print(f"Majority bounding box for second camera frame is {majority_value_cam2} with {majority_occurence_cam2} occurrences.")

            # Generate more points
            # additional_coords_image1 = []
            # additional_coords_image2 = []
            # for i in range(10):
            #     additional_coords_image1.append(((random.randint(majority_value_cam1[0], majority_value_cam1[2])), random.randint(majority_value_cam1[1], majority_value_cam1[3])))
            #     additional_coords_image2.append(((random.randint(majority_value_cam2[0], majority_value_cam2[2])), random.randint(majority_value_cam2[1], majority_value_cam2[3])))
            
            # print(additional_coords_image1)
            # print(additional_coords_image2)

            print("Matching keypoints...")
            bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

            matches = bf.match(image1_descriptors, image2_descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

            print(f"Number of matches for frame {frame_index}: {len(matches)}")
            print(matches)

            img3 = cv.drawMatches(image1, image1_keypoints, image2, image2_keypoints, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            matched_keypoints_image1 = [image1_keypoints[m.queryIdx] for m in matches]
            matched_keypoints_image2 = [image2_keypoints[m.trainIdx] for m in matches]
            matched_keypoints_image1_coordinates = cv.KeyPoint_convert(matched_keypoints_image1)
            matched_keypoints_image2_coordinates = cv.KeyPoint_convert(matched_keypoints_image2)

            coords = rec.triangulate_points(matched_keypoints_image1_coordinates, matched_keypoints_image2_coordinates, rec.proj_matrix1, rec.proj_matrix2)
            rec.plot3d(coords)

            break

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

                # new_mat = np.array([[image1_keypoint], [image1_keypoint_bbox], [image2_keypoint], [image2_keypoint_bbox]])
                # new_mat = np.asarray(new_mat)
                # large_mat.append(new_mat)

            # frame_bbox_occurences = large_mat[:,1]
            # print(bbox_dict)
            values = list(bbox_dict.values())
            # print(values)
            # plt.hist(list(bbox_dict.keys()), bins=len(bbox_dict.keys()), edgecolor='black')
            # plt.show()


            # majority_bbox = max(zip(bbox_dict.values(), bbox_dict.keys()))[1]
            # for i in range(len(matched_keypoints_image2)-1):
            #     if keypoints_bbox_map2[matched_keypoints_image2[i]] == majority_bbox:
            #         pass
            #     else:
            #         print("Invalid match, removing from matches")
            #         print(keypoints_bbox_map2[matched_keypoints_image2[i]])
                    
            #         if matched_keypoints_image2[i] is not None:
            #             matched_keypoints_image1.pop(i)
            #             matched_keypoints_image2.pop(i)


            # for feature in image1_features:
            #     if feature.coordinate in matched_keypoints_image1_coordinates:
            #         feature.matched = True
            #         feature.matched_feature = matched_keypoints_image2_coordinates[np.where(matched_keypoints_image1_coordinates == feature.coordinate)]
            #         # feature.matched_bbox = image2_features[np.where(image2_features)]


            #         np.where([(2.01, 3), (2.01, 4)])
            #         print(feature.matched_feature)

            # for feature in image2_features:
            #     if feature.coordinate in matched_keypoints_image2_coordinates:
            #         feature.matched = True
            #         feature.matched_feature = matched_keypoints_image1_coordinates[np.where(matched_keypoints_image2_coordinates == feature.coordinate)]


            # for feature1 in image1_features:
            #     for feature2 in image2_features:
            #         if feature1.matched_feature == feature2.coordinate:
            #             feature1.matched_bbox = feature2.bbox
            #             feature2.matched_bbox = feature1.bbox
                    
            #         print(feature1.coordinate, feature1.bbox, feature1.matched_bbox)
            #         print(feature2.coordinate, feature2.bbox, feature2.matched_bbox)
            # Get keypoint match pairs to begin outlier rejection

            # show = False

            # if frame_index % 10 == 0:
            #     plt.imsave(f'/Users/keonm/Desktop/Bird_Project/Moradi_Aviary_Project/initially_matched_features{frame_index}.png', img3)
            #     show = False

            # image1_filtered_keypoints, image2_filtered_keypoints = cbfm.get_keypoint_match_pairs(
            #     image1_keypoints, image2_keypoints, matches, frame_index,
            #     image1_paths[frame_index], image2_paths[frame_index],
            #     image1_selected_landmarks, image2_selected_landmarks, show=show,
            #     )
            


    #         try:
    #             percent_kp1_accepted = (len(image1_filtered_keypoints) / len(image1_keypoints)) * 100
    #             percent_kp_1_rejected = 100 - percent_kp1_accepted
    #         except:
    #             percent_kp1_accepted = 0
    #             percent_kp_1_rejected = 0

    #         try:  
    #             percent_kp2_accepted = (len(image2_filtered_keypoints) / len(image2_keypoints)) * 100
    #             percent_kp_2_rejected = 100 - percent_kp2_accepted
    #         except:
    #             percent_kp2_accepted = 0
    #             percent_kp_2_rejected = 0

            




    #         frame_rejection_rate = (percent_kp_1_rejected + percent_kp_2_rejected) / 2


    #         if frame_rejection_rate != 100:

    #             rejections_over_time.append(frame_rejection_rate)
    #             frame_counter.append(frame_index)

    #         total_kp_count += len(image1_keypoints)
    #         total_kp_count += len(image2_keypoints)
            
    #         total_filtered_kp_count += len(image1_filtered_keypoints)
    #         total_filtered_kp_count += len(image2_filtered_keypoints)

    #         print(f'Percent of image1 keypoints accepted: {percent_kp1_accepted}')
    #         print(f'Percent of image2 keypoints accepted: {percent_kp2_accepted}')

    #         print(f"Number of matches for frame {frame_index}: {len(matches)}")

    # print('Processing completed')

    # kp_counts_img1 = np.array(kp_counts_img1)
    # kp_counts_img2 = np.array(kp_counts_img2)

    # print(f'Minimum number of keypoints in image1: {np.min(kp_counts_img1)}')
    # print(f'Minimum number of keypoints in image2: {np.min(kp_counts_img2)}')

    # print(f'Maximum number of keypoints in image1: {np.max(kp_counts_img1)}')
    # print(f'Maximum number of keypoints in image2: {np.max(kp_counts_img2)}')

    # print(f'Average number of keypoints in image1: {np.average(kp_counts_img1)}')
    # print(f'Average number of keypoints in image2: {np.average(kp_counts_img2)}')

    # print(f'Standard deviation of number of keypoints in image1: {np.std(kp_counts_img1)}')
    # print(f'Standard deviation of number of keypoints in image2: {np.std(kp_counts_img2)}')

    # print(f'Average rejection rate over interval: {np.average(rejections_over_time)}')
    # print(f'Standard deviation of rejection rate over interval: {np.std(rejections_over_time)}')

    # fig, ax = plt.subplots()
    # ax.scatter(frame_counter, rejections_over_time)

    # ax.set_xlabel('Frame Number')  # X-axis label
    # ax.set_ylabel('Rejections Over Time (%)')  # Y-axis label
    # ax.set_title('Rejection Rate Over Time vs Frame Number')
    # # plt.show()
    # fig.savefig('error_graph.png')

if __name__ == "__main__":
    for camera1, camera2 in camera_pairs:
        main(camera1, camera2, start_frame=3, end_frame=7, select_new_landmarks=False, get_keypoints=True)

