import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import os
import KeypointExtractor as kp
import numpy as np
import ContextBasedFeatureMatching as cbfm
# from reconstruction import triangulate_points

image_path = '/home/golnaz/Moradi_Aviary_Project/'
camera_pairs = [(1, 3), (1, 5)]
camera_names = ['gopro1', 'gopro2', 'gopro3', 'gopro4', 'gopro5']
camera_frames_directories = [
    image_path+'pipeline/video_frames/GoPro1_Encl4_03152024_1',
    image_path+'pipeline/video_frames/GoPro2_Encl4_03152024_1',
    image_path+'pipeline/video_frames/GoPro3_Encl4_03152024_1',
    image_path+'pipeline/video_frames/GoPro4_Encl4_03152024_1',
    image_path+'pipeline/video_frames/GoPro5_Encl4_03152024_1',
]

first_frames_paths = [
     image_path+'pipeline/video_frames/GoPro1_Encl4_03152024_1/GoPro1_Encl4_03152024_1_frame_0000.jpg',
     image_path+'pipeline/video_frames/GoPro2_Encl4_03152024_1/GoPro2_Encl4_03152024_1_frame_0000.jpg',
    image_path+ 'pipeline/video_frames/GoPro3_Encl4_03152024_1/GoPro3_Encl4_03152024_1_frame_0000.jpg',
     image_path+'pipeline/video_frames/GoPro4_Encl4_03152024_1/GoPro4_Encl4_03152024_1_frame_0000.jpg',
     image_path+'pipeline/video_frames/GoPro5_Encl4_03152024_1/GoPro5_Encl4_03152024_1_frame_0000.jpg',
]

def rindex(lst, value):
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] == value:
            return i
    return -1

def locate_landmarks_in_images(image_paths, camera_num, save_landmarks=True):

    output_directory = f'{camera_frames_directories[camera_num-1]}/'
    landmarks = {
    "left-bottom1": [],
    "left-bottom2": [],
    "left-bottom3": [],
    "left-middle1": [],
    "left-middle2": [],
    "left-middle3": [],
    "left-top1": [],
    "left-top2": [],
    "left-top3": [],
    "right-bottom1": [],
    "right-bottom2": [],
    "right-bottom3": [],
    "right-middle1": [],
    "right-middle2": [],
    "right-middle3": [],
    "right-top1": [],
    "right-top2": [],
    "right-top3": [],
    "left-door1": [],
    "left-door2": [],
    "left-door3": [],
    "right-door1": [],
    "right-door2": [],
    "right-door3": [],
    "left-back1": [],
    "left-back2": [],
    "left-back3": [],
    "right-back1": [],
    "right-back2": [],
    "right-back3": [],
    "center-back1": [],
    "center-back2": [],
    "center-back3": [],
    }

    for image_path in image_paths:
        image = cv.imread(image_path)
        cv.imshow('Camera view', image)

        for landmark in landmarks:
            print(f'Current landmark: {landmark}')

            # Define a flag to indicate if a click has occurred
            click_occurred = False

            # Define a mouse callback function
            def mouse_callback(event, x, y, flags, param):
                nonlocal click_occurred
                if event == cv.EVENT_LBUTTONDOWN:
                    print(f"Clicked coordinates for {landmark}: ({x}, {y})")
                    landmarks[landmark].append([x, y])
                    click_occurred = True

            # Set mouse callback function for the window
            cv.setMouseCallback('Camera view', mouse_callback)

            # Wait until a click occurs (handled by the callback)
            while not click_occurred:
                key = cv.waitKey(1) & 0xFF
                if key == ord('x'):
                    print(f'Skipped landmark {landmark}')
                    landmarks[landmark].append('none')
                    break

            # Reset click_occurred for the next landmark
            click_occurred = False

        cv.destroyAllWindows()
    
    if save_landmarks:
        np.save(f'{output_directory}/selected_landmarks.npy', landmarks)
        print("Saved landmarks")

    return landmarks

def main(camera1_number, camera2_number, start_frame=3, end_frame=7, select_new_landmarks=False, get_keypoints=True):
    
    # image1_selected_landmarks = None
    # image2_selected_landmarks = None


    # if get_keypoints:
        # image1_selected_landmarks = locate_landmarks_in_images(first_frames_paths[camera1_number-1], camera_num=camera1_number, save_landmarks=True)
        # image2_selected_landmarks = locate_landmarks_in_images(first_frames_paths[camera2_number-1], camera_num=camera2_number, save_landmarks=True)


    # else:
    #     image1_selected_landmarks = 
    #     image2_selected_landmarks = 

    # image1_selected_landmarks = locate_landmarks_in_images([first_frames_paths[camera1_number-1]], camera_num=camera1_number, save_landmarks=True)
    # image2_selected_landmarks = locate_landmarks_in_images([first_frames_paths[camera2_number-1]], camera_num=camera2_number, save_landmarks=True)

    image1_selected_landmarks = cbfm.select_landmarks(first_frames_paths[camera1_number-1], cbfm.landmark_names)
    image2_selected_landmarks = cbfm.select_landmarks(first_frames_paths[camera2_number-1], cbfm.landmark_names)


    detections1_path = os.path.join(image_path,f'pipeline/detections/clean_csvs/gopro{camera1_number}_clean.csv')
    detections2_path = os.path.join(image_path,f'pipeline/detections/clean_csvs/gopro{camera2_number}_clean.csv')

    detections1 = pd.read_csv(detections1_path)
    detections2 = pd.read_csv(detections2_path)

    # if get_keypoints:
    #     gp.get_keypoints(GoPro_frames_folder=camera_frames_directories[camera1_number], GoPro_detections_path=detections1_path)
    #     gp.get_keypoints(GoPro_frames_folder=camera_frames_directories[camera2_number], GoPro_detections_path=detections2_path)


    relevant_detections1 = detections1[(detections1["frame_index"] >= start_frame) & (detections1["frame_index"] <= end_frame)]
    relevant_detections2 = detections2[(detections2["frame_index"] >= start_frame) & (detections2["frame_index"] <= end_frame)]


    common_values1 = set(relevant_detections2["frame_index"].unique())
    common_values2 = set(relevant_detections1["frame_index"].unique())

    final_relevant_detections1 = relevant_detections1[relevant_detections1["frame_index"].isin(common_values1)]
    final_relevant_detections2 = relevant_detections2[relevant_detections2["frame_index"].isin(common_values2)]

    image1_frame_bounded_boxes = {
        num: [] for num in common_values1
    }

    image2_frame_bounded_boxes = {
        num: [] for num in common_values1
    }


    for index, detection in final_relevant_detections1.iterrows():
        xmin, ymin, xmax, ymax = detection['x1'], detection['y1'], detection['x2'], detection['y2']
        image1_frame_bounded_boxes[detection['frame_index']].append((xmin, ymin, xmax, ymax))

    for index, detection in final_relevant_detections2.iterrows():
        xmin, ymin, xmax, ymax = detection['x1'], detection['y1'], detection['x2'], detection['y2']
        image2_frame_bounded_boxes[detection['frame_index']].append((xmin, ymin, xmax, ymax))

    print('Obtained bounded box coordinates for all detections')

    #Load relevant frames from both camera views
        
    image1_paths = {
        num: None for num in common_values1
    }
        
    image2_paths = {
        num: None for num in common_values1
    }

    for frame_num in common_values1:
        frame_label = ''
        for i in range(4 - len([int(i) for i in str(frame_num)])):
            frame_label += '0'

        frame_label += str(frame_num)
        print(frame_label)
        image1_paths[frame_num] = f'{camera_frames_directories[camera1_number-1]}/{os.path.basename(camera_frames_directories[camera1_number-1])}_frame_{frame_label}.jpg'
        image2_paths[frame_num] = f'{camera_frames_directories[camera2_number-1]}/{os.path.basename(camera_frames_directories[camera2_number-1])}_frame_{frame_label}.jpg'

    print('Beginning keypoint extraction')

    for frame_index in common_values1:

        image1 = cv.imread(image1_paths[frame_index])
        image2 = cv.imread(image2_paths[frame_index])

        image1_gray = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
        image1_edges = cv.Canny(image1_gray, 100, 200)

        image2_gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
        image2_edges = cv.Canny(image2_gray, 100, 200)
        # image1_path = image1_paths[frame_index]
        # image2_path = image2_paths[frame_index]

        image1_mask = np.zeros_like(image1[:, :, 0])
        image1_new_mask = np.zeros_like(image1[:, :, 0])

        image2_mask = np.zeros_like(image2[:, :, 0])
        image2_new_mask = np.zeros_like(image2[:, :, 0])

        # image1_bbox_coords = 

        bbox_dict = {
                count: None for count in range(len(image1_frame_bounded_boxes[frame_index]))
            }

        print("Obtaining masks for first camera frames")

        count = 0
        for bbox_coords in image1_frame_bounded_boxes[frame_index]:


            xmin, ymin, xmax, ymax = int(bbox_coords[0]), int(bbox_coords[1]), int(bbox_coords[2]), int(bbox_coords[3])
            
            bbox_center_coords = ((xmin+xmax)/2, (ymin+ymax)/2)

            

            bbox_dict[count] = bbox_center_coords

            count += 1
            
            cv.rectangle(image1_mask, (xmin, ymin), (xmax, ymax), 255, thickness=cv.FILLED)  

            region_inside_edges = cv.bitwise_and(image1_edges, image1_edges, mask=image1_mask)

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
                            image1_new_mask[y, fill] = 255
                    elif end == start:
                        image1_new_mask[y, start] = 255

        print("Obtaining masks for second camera frames")

        for bbox_coords in image2_frame_bounded_boxes[frame_index]:

            xmin, ymin, xmax, ymax = int(bbox_coords[0]), int(bbox_coords[1]), int(bbox_coords[2]), int(bbox_coords[3])
            cv.rectangle(image2_mask, (xmin, ymin), (xmax, ymax), 255, thickness=cv.FILLED)  

            region_inside_edges = cv.bitwise_and(image2_edges, image2_edges, mask=image2_mask)

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
                            image2_new_mask[y, fill] = 255
                    elif end == start:
                        image2_new_mask[y, start] = 255

        sift = cv.SIFT_create(nfeatures=5000)
        #sift = cv.xfeatures2d.SURF_create(nfeatures=5000)
        print('Obtaining keypoints for first camera frames')
        image1_keypoints, image1_descriptors = sift.detectAndCompute(image1_new_mask, None)

        print('Obtaining keypoints for second camera frames')
        image2_keypoints, image2_descriptors = sift.detectAndCompute(image2_new_mask, None)

        print('Drawing keypoints over images')
        image1_with_keypoints =cv.drawKeypoints(image1, image1_keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        image2_with_keypoints = cv.drawKeypoints(image2, image2_keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        image1_keypoints_serialized = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in image1_keypoints]
        image2_keypoints_serialized = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in image2_keypoints]


        # print("Image 1 Descriptors Type:", image1_descriptors.dtype)
        # print("Image 2 Descriptors Type:", image2_descriptors.dtype)

        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

        
        matches = bf.match(image1_descriptors, image2_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        img3 = cv.drawMatches(image1, image1_keypoints, image2, image2_keypoints, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Get keypoint match pairs to begin outlier rejection

        image1_filtered_keypoints, image2_filtered_keypoints = cbfm.get_keypoint_match_pairs(
            image1_keypoints, image2_keypoints, matches, 
            image1_paths[frame_index], image2_paths[frame_index],
            image1_selected_landmarks, image2_selected_landmarks, show=True
            )
        
        percent_kp1_accepted = (len(image1_filtered_keypoints) / len(image1_keypoints)) * 100
        percent_kp2_accepted = (len(image2_filtered_keypoints) / len(image2_keypoints)) * 100

        print(f'Percent of image1 keypoints accepted: {percent_kp1_accepted}')
        print(f'Percent of image2 keypoints accepted: {percent_kp2_accepted}')
        
        image1_keypoints_coordinates = cv.KeyPoint_convert(image1_filtered_keypoints)
        image2_keypoints_coordinates = cv.KeyPoint_convert(image2_filtered_keypoints)
        

main(camera1_number=3, camera2_number=4, start_frame=3, end_frame=7, get_keypoints=True)
