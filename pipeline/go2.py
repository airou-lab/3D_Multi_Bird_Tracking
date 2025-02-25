import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import ContextBasedFeatureMatching as cbfm
image_path = '/home/golnaz/Moradi_Aviary_Project/'
camera_pairs = [(3, 4)]
camera_names = ['gopro1', 'gopro2', 'gopro3', 'gopro4', 'gopro5']
camera_frames_directories = [
    'video_frames/GoPro1_Encl4_03152024_1',
    'video_frames/GoPro2_Encl4_03152024_1',
    'video_frames/GoPro3_Encl4_03152024_1',
    'video_frames/GoPro4_Encl4_03152024_1',
    'video_frames/GoPro5_Encl4_03152024_1',
]

first_frames_paths = [
    'video_frames/GoPro1_Encl4_03152024_1/GoPro1_Encl4_03152024_1_frame_0000.jpg',
    'video_frames/GoPro2_Encl4_03152024_1/GoPro2_Encl4_03152024_1_frame_0000.jpg',
    'video_frames/GoPro3_Encl4_03152024_1/GoPro3_Encl4_03152024_1_frame_0000.jpg',
    'video_frames/GoPro4_Encl4_03152024_1/GoPro4_Encl4_03152024_1_frame_0000.jpg',
    'video_frames/GoPro5_Encl4_03152024_1/GoPro5_Encl4_03152024_1_frame_0000.jpg',
]

def rindex(lst, value):
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] == value:
            return i
    return -1

def locate_landmarks_in_images(image_paths, camera_num, save_landmarks=True):
    output_directory = f'{camera_frames_directories[camera_num-1]}/'
    landmarks = {name: [] for name in [
        "left-bottom1", "left-bottom2", "left-bottom3", "left-middle1", "left-middle2", "left-middle3",
        "left-top1", "left-top2", "left-top3", "right-bottom1", "right-bottom2", "right-bottom3",
        "right-middle1", "right-middle2", "right-middle3", "right-top1", "right-top2", "right-top3",
        "left-door1", "left-door2", "left-door3", "right-door1", "right-door2", "right-door3",
        "left-back1", "left-back2", "left-back3", "right-back1", "right-back2", "right-back3",
        "center-back1", "center-back2", "center-back3"
    ]}
    
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

def main(camera1_number, camera2_number, start_frame=1, end_frame=3, select_new_landmarks=False, get_keypoints=True):
    image1_selected_landmarks = cbfm.select_landmarks(first_frames_paths[camera1_number-1], cbfm.landmark_names)
    image2_selected_landmarks = cbfm.select_landmarks(first_frames_paths[camera2_number-1], cbfm.landmark_names)

   



    detections1_path = os.path.join(image_path,f'pipeline/detections/clean_csvs/gopro{camera1_number}_clean.csv')
    detections2_path = os.path.join(image_path,f'pipeline/detections/clean_csvs/gopro{camera2_number}_clean.csv')
    detections1 = pd.read_csv(detections1_path)
    detections2 = pd.read_csv(detections2_path)

    relevant_detections1 = detections1[(detections1["frame_index"] >= start_frame) & (detections1["frame_index"] <= end_frame)]
    relevant_detections2 = detections2[(detections2["frame_index"] >= start_frame) & (detections2["frame_index"] <= end_frame)]

    common_values = set(relevant_detections1["frame_index"]).intersection(set(relevant_detections2["frame_index"]))

    image1_frame_bounded_boxes = {num: [] for num in common_values}
    image2_frame_bounded_boxes = {num: [] for num in common_values}

    for index, detection in relevant_detections1.iterrows():
        xmin, ymin, xmax, ymax = detection['x1'], detection['y1'], detection['x2'], detection['y2']
        image1_frame_bounded_boxes[detection['frame_index']].append((xmin, ymin, xmax, ymax))

    for index, detection in relevant_detections2.iterrows():
        xmin, ymin, xmax, ymax = detection['x1'], detection['y1'], detection['x2'], detection['y2']
        image2_frame_bounded_boxes[detection['frame_index']].append((xmin, ymin, xmax, ymax))

    print('Obtained bounded box coordinates for all detections')

    image1_paths = {num: f'{camera_frames_directories[camera1_number-1]}/{os.path.basename(camera_frames_directories[camera1_number-1])}_frame_000{num}.jpg' for num in common_values}
    image2_paths = {num: f'{camera_frames_directories[camera2_number-1]}/{os.path.basename(camera_frames_directories[camera2_number-1])}_frame_000{num}.jpg' for num in common_values}

    print('Beginning keypoint extraction')

    for frame_index in common_values:
        image1 = cv.imread(image1_paths[frame_index])
        image2 = cv.imread(image2_paths[frame_index])

        image1_gray = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
        image1_edges = cv.Canny(image1_gray, 100, 200)

        image2_gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
        image2_edges = cv.Canny(image2_gray, 100, 200)

        image1_mask = np.zeros_like(image1[:, :, 0])
        image2_mask = np.zeros_like(image2[:, :, 0])

        for bbox_coords in image1_frame_bounded_boxes[frame_index]:
            xmin, ymin, xmax, ymax = int(bbox_coords[0]), int(bbox_coords[1]), int(bbox_coords[2]), int(bbox_coords[3])
            cv.rectangle(image1_mask, (xmin, ymin), (xmax, ymax), 255, thickness=cv.FILLED)  
            region_inside_edges = cv.bitwise_and(image1_edges, image1_edges, mask=image1_mask)
            for y in range(region_inside_edges.shape[0]):
                row = region_inside_edges[y]
                start = np.argmax(row != 0)
                end = rindex(row, 255)
                if start >= 0 and end > start:
                    image1_mask[y, start:end + 1] = 255

        for bbox_coords in image2_frame_bounded_boxes[frame_index]:
            xmin, ymin, xmax, ymax = int(bbox_coords[0]), int(bbox_coords[1]), int(bbox_coords[2]), int(bbox_coords[3])
            cv.rectangle(image2_mask, (xmin, ymin), (xmax, ymax), 255, thickness=cv.FILLED)  
            region_inside_edges = cv.bitwise_and(image2_edges, image2_edges, mask=image2_mask)
            for y in range(region_inside_edges.shape[0]):
                row = region_inside_edges[y]
                start = np.argmax(row != 0)
                end = rindex(row, 255)
                if start >= 0 and end > start:
                    image2_mask[y, start:end + 1] = 255

        sift = cv.SIFT_create(nfeatures=5000)

        print('Obtaining keypoints for first camera frame')
        image1_keypoints, image1_descriptors = sift.detectAndCompute(image1_mask, None)

        print('Obtaining keypoints for second camera frame')
        image2_keypoints, image2_descriptors = sift.detectAndCompute(image2_mask, None)

        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        matches = bf.match(image1_descriptors, image2_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        print('Keypoints match pairs')
        image1_filtered_keypoints_coords, image2_filtered_keypoints_coords = cbfm.get_keypoint_match_pairs(
            image1_keypoints, image2_keypoints, matches,
            image1_paths[frame_index], image2_paths[frame_index],
            image1_selected_landmarks, image2_selected_landmarks, show=True
        )

        def triangulate_points(image1_filtered_keypoints_coords, image2_filtered_keypoints_coords):
            # points1_homogeneous = cv.convertPointsToHomogeneous(np.array(image1_filtered_keypoints)).reshape(-1, 4).T
            # points2_homogeneous = cv.convertPointsToHomogeneous(np.array(image2_filtered_keypoints)).reshape(-1, 4).T

            # keypoints1_coordinates = cv.KeyPoint_convert(image1_filtered_keypoints)
            # keypoints2_coordinates = cv.KeyPoint_convert(image2_filtered_keypoints)

            # points1_homogeneous = cv.convertPointsToHomogeneous(keypoints1_coordinates)
            # points2_homogeneous = cv.convertPointsToHomogeneous(keypoints2_coordinates)

            # points1 = np.array([kp.pt for kp in image1_filtered_keypoints], dtype=np.float32)
            # points2 = np.array([kp.pt for kp in image2_filtered_keypoints], dtype=np.float32)
    
            # Convert points to homogeneous coordinates
            points1_homogeneous = cv.convertPointsToHomogeneous(image1_filtered_keypoints_coords)
            points2_homogeneous = cv.convertPointsToHomogeneous(image2_filtered_keypoints_coords)

            P1 = np.array([
                [1.0, 0.0, 0.0, -0.84914447],
                [0.0, 1.0, 0.0, 0.52815994],
                [0.0, 0.0, 1.0, -0.0008683]
            ])
            
            P2 = np.array([
                [1.0, 7.38870490e-06, -2.64861543e-07, -9.77816546e-01],
                [-7.38870467e-06, 1.0, 8.40792419e-07, 2.09461613e-01],
                [2.64867756e-07, -8.40790461e-07, 1.0, -7.97397492e-04]
            ])

            print(f'Number of points in first image: {len(P1)}')
            print(f'Number of points in second image: {len(P2)}')

            print(f'Number of homogenous first points: {len(points1_homogeneous)}, dim: {len(points1_homogeneous[0])}')
            print(f'Number of homogenous second points: {len(points2_homogeneous)}, dim: {len(points2_homogeneous[0])}')

            points4D_homogeneous = cv.triangulatePoints(P1, P2, points1_homogeneous.T, points2_homogeneous.T)
            points3D = points4D_homogeneous[:3] / points4D_homogeneous[3]
            return points3D.T

        if len(image1_filtered_keypoints_coords) > 0 and len(image2_filtered_keypoints_coords) > 0:
            points3D = triangulate_points(image1_filtered_keypoints_coords, image2_filtered_keypoints_coords)
            plot_3d_points(points3D)
        else:
            print("No matching keypoints for triangulation")

    print('End of keypoint matching and triangulation')

def plot_3d_points(points3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2])
    plt.show()

if __name__ == "__main__":
    for camera1, camera2 in camera_pairs:
        main(camera1, camera2)
