import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.path import Path
from shapely.geometry import Polygon
from PIL import Image, ImageDraw

landmark_names = [
    "left-bottom1", "left-bottom2", "left-bottom3",
    "left-middle1", "left-middle2", "left-middle3",
    "left-top1", "left-top2", "left-top3",
    "right-bottom1", "right-bottom2", "right-bottom3",
    "right-middle1", "right-middle2", "right-middle3",
    "right-top1", "right-top2", "right-top3",
    "left-door1", "left-door2", "left-door3",
    "right-door1", "right-door2", "right-door3",
    "left-back1", "left-back2", "left-back3",
    "right-back1", "right-back2", "right-back3",
    "center-back1", "center-back2", "center-back3"
]

def load_landmarks(landmark_name):
    return np.load(f'landmarks/{landmark_name}', allow_pickle=True).item()

def number_landmarks(landmark_names, csv_file_name):
    # Prepare data for CSV
    data = []
    for index, landmark in enumerate(landmark_names, start=1):
        data.append([index, landmark])
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Index', 'Landmark Name'])
    
    # Save DataFrame to CSV
    df.to_csv(csv_file_name, index=False)
    print(f"Landmarks have been numbered and saved to {csv_file_name}")


def select_landmarks(image_path, landmark_names):
    # Initialize dictionary to store coordinates
    selected_landmarks = {name: None for name in landmark_names}
    
    # Load the image and prepare a copy for annotation
    image = cv2.imread(image_path)
    clone = image.copy()

    # Define the current landmark index
    current_landmark_index = 0

    def handle_key(key):
        nonlocal current_landmark_index, selected_landmarks, clone
        if key == ord('x'):
            # Move to the next landmark when 'x' is pressed
            # current_landmark_index += 1
            if current_landmark_index < len(landmark_names):
                # Update the current landmark being selected
                landmark_name = landmark_names[current_landmark_index]
                print(f"Skipping {landmark_name}")
                return True
                # clone = image.copy()  # Reset the clone for next selection
            else:
                print("No more landmarks to skip")
        elif key == ord('q') or current_landmark_index >= len(landmark_names):
            # Exit if 'q' is pressed or all landmarks have been processed
            return False
        return True

    def click_and_select(event, x, y, flags, param):
        nonlocal current_landmark_index, selected_landmarks, clone
        
        if event == cv2.EVENT_LBUTTONDOWN:

            # current_landmark_index += 1
            if current_landmark_index >= len(landmark_names):
                print("All landmarks have been processed.")
                return
            
            # Draw a circle on the clone image where the click occurred
            cv2.circle(clone, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow("Select Landmarks (press x to skip)", clone)

            # Assign the coordinates to the current landmark
            landmark_name = landmark_names[current_landmark_index]
            selected_landmarks[landmark_name] = (x, y)
            print(f"Selected {landmark_name} at coordinates ({x}, {y})")
            current_landmark_index += 1

            key = cv2.waitKey(0) & 0xFF
            if handle_key(key):
                current_landmark_index += 1
                landmark_name = landmark_names[current_landmark_index]
                return
        
            
            # # Assign the coordinates to the current landmark
            # landmark_name = landmark_names[current_landmark_index]
            # selected_landmarks[landmark_name] = (x, y)
            # print(f"Selected {landmark_name} at coordinates ({x}, {y})")
            
            # Move to the next landmark
            # current_landmark_index += 1
            if current_landmark_index < len(landmark_names):
                # Update the current landmark being selected
                landmark_name = landmark_names[current_landmark_index]
                # clone = image.copy()  # Reset the clone for next selection
            else:
                # If all landmarks have been selected, exit the loop
                print("All landmarks have been selected")

    # Create a window and set the mouse callback function
    cv2.namedWindow("Select Landmarks (press x to skip)")
    cv2.setMouseCallback("Select Landmarks (press x to skip)", click_and_select)

    # Display the image and wait for user input
    while True:
        cv2.imshow("Select Landmarks (press x to skip)", clone)
        key = cv2.waitKey(1) & 0xFF
        if not handle_key(key):
            break

    cv2.destroyAllWindows()
    
    return selected_landmarks


def load_birds(csv_path, pad_size=0):
    # Load bird coordinates from CSV file
    use_cols = ['xmin', 'ymin', 'xmax', 'ymax']
    birds_csv = pd.read_csv(csv_path, usecols=use_cols)
    bird_coords = []

    for _, row in birds_csv.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        
        # Calculate center of bird coordinates
        x_mid = int((xmin + xmax) / 2)
        y_mid = int((ymin + ymax) / 2)
        
        # Adjust coordinates to account for padding
        x_mid += pad_size 
        y_mid += pad_size
        
        bird_coords.append((x_mid, y_mid))

    return bird_coords

def number_birds(bird_coords, csv_file_name):
        # Prepare data for CSV
    data = []
    for index, bird in enumerate(bird_coords, start=1):
        data.append([index, bird])
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Index', 'Bird_coords'])
    
    # Save DataFrame to CSV
    df.to_csv(csv_file_name, index=False)
    print(f"Birds have been numbered and saved to {csv_file_name}")

def bird_closest_landmarks(image_path, selected_landmarks, bird_coords):
    closest_landmarks = []
    closest_distances = []
    bird_number = 1
    image = cv2.imread(image_path)

    for bird_coord in bird_coords:
        bird_center = np.array(bird_coord)
        
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

        cv2.line(image, bird_coord, selected_landmarks[closest_landmark], (0, 255, 0), 2)
        cv2.circle(image, bird_coord, 5, (0, 0, 255), -1)
        cv2.circle(image, selected_landmarks[closest_landmark], 5, (255, 0, 0), -1)
        cv2.putText(image, f'{closest_landmark}', selected_landmarks[closest_landmark], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f'Bird {bird_number}: {min_dist:.2f}', bird_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        print(f'Bird {bird_number} at location {bird_coord} is closest to landmark {closest_landmark} at a distance of {min_dist}.')
        bird_number += 1
    
    cv2.imshow("Birds and Landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return closest_landmarks, closest_distances

# def print_bird_landmark_pairs(selected_landmarks, bird_coords):
#     closest_landmarks, closest_distances = bird_closest_landmarks(selected_landmarks=select_landmarks, bird_coords=bird_coords)

#     for index, bird in enumerate(bird_coords):
#         print('Bird {index} at location {bird} is closest to landmark {closest_landmarks[index]} at a distance of {closest_distances[index]}.')
        

if __name__ == "__main__":
    image_path = 'images/frame_0615_1.jpg'
    birds_path = 'bounded_boxes/cam_frame615_boxes.csv'
    number_birds(birds_path, 'bird_coords/cam_frame615_birds.csv')
    bird_coords = load_birds('bounded_boxes/cam1_frame615_boxes.csv')
    selected_landmarks = select_landmarks(image_path=image_path, landmark_names=landmark_names)
    bird_closest_landmarks(image_path=image_path, selected_landmarks=selected_landmarks, bird_coords=bird_coords)