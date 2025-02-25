import argparse
import pandas as pd
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

def rindex(lst, value):
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] == value:
            return i
    return -1

def load_image(image_path):
    return cv.imread(image_path)

def load_bounded_boxes(bounded_box_path):
    return pd.read_csv(bounded_box_path)

def save_keypoints(keypoints, image_name, output_directory_path):
    keypoints_list = [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]
    keypoints_array = np.array(keypoints_list, dtype=np.float32)
    output_path = f'{output_directory_path}{image_name[:-4]}_keypoints.npy'
    np.save(output_path, keypoints_array)

def save_descriptors(descriptors, image_name, output_directory_path):
    output_path = f'{output_directory_path}{image_name[:-4]}_descriptors.npy'
    np.save(output_path, descriptors)

def extract_keypoints(image_path, image_name, bounded_box_path, output_directory_path, show_keypoints, show_mask, save=True, save_img_with_keypoints=False):
    img = load_image(image_path)
    boxes = load_bounded_boxes(bounded_box_path)
    new_mask = np.zeros_like(img[:, :, 0])

    for index, row in boxes.iterrows():
        xmin, ymin, xmax, ymax = int(row['x1']), int(row['y2']), int(row['x2']), int(row['y2'])
        
        mask = np.zeros_like(img[:, :, 0]) 
        cv.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, thickness=cv.FILLED)  
        
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        edges = cv.Canny(gray_img, 100, 200)
        
        region_inside_edges = cv.bitwise_and(edges, edges, mask=mask)

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
                        new_mask[y, fill] = 255
                elif end == start:
                    new_mask[y, start] = 255

    sift = cv.SIFT_create(nfeatures=5000)

    keypoints, descriptors = sift.detectAndCompute(new_mask, None)

    img_with_keypoints = cv.drawKeypoints(img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    keypoints_serialized = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

    if save:
        
        try:
            save_keypoints(keypoints, image_name, output_directory_path=output_directory_path)
            print(f'Successfully saved keypoints to {output_directory_path}')

            save_descriptors(descriptors, image_name, output_directory_path=output_directory_path)
            print(f'Successfully saved descriptors to {output_directory_path}')
            
        except Exception as e:
            print(f'An error occurred: {e}')

    if show_keypoints:
        plt.imshow(cv.cvtColor(img_with_keypoints, cv.COLOR_BGR2RGB))
        plt.title('Image with SIFT Keypoints')
        plt.show()
    
    if show_mask:
        plt.imshow(new_mask, cmap='gray')
        plt.title('Image with mask')
        plt.show() 


    if save_img_with_keypoints:
        image_filename = os.path.basename(image_name)
        image_save_path = os.path.join('images_with_keypoints', image_filename)

        try:
            if cv.imwrite(image_save_path, img_with_keypoints):
                print(f'Successfully saved image to {image_save_path}')
            else:
                print(f'Failed to save image to {image_save_path}')

        except Exception as e:
            print(f'An error occurred: {e}')


    return keypoints, descriptors


# Use like: python KeypointExtractor.py "frame_0615_1.jpg" "cam1_frame615_boxes.csv" --show_keypoints --show_mask




# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Extract keypoints from images within bounding boxes')
#     parser.add_argument('image_name', help='Name of the image located in the "images" folder')
#     parser.add_argument('bounding_box_name', help='Name of the bounding box file located in the "bounded_boxes" folder')
#     parser.add_argument('--show_keypoints', action='store_true', help='Display the image with keypoints')
#     parser.add_argument('--show_mask', action='store_true', help='Display the image with mask')
#     args = parser.parse_args()

#     extract_keypoints(args.image_name, args.bounding_box_name, args.show_keypoints, args.show_mask, save=True)
