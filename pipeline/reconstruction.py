import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Function to triangulate 3D points from corresponding 2D points in two camera views
def triangulate_points(pts1, pts2, proj_matrix1, proj_matrix2):
    print("Points1: ", pts1)
    print("Points2: ", pts2)
    pts1 = np.array(pts1, dtype='float32').T  # Convert points from the first camera to numpy array and transpose
    pts2 = np.array(pts2, dtype='float32').T  # Convert points from the second camera to numpy array and transpose
    points_4d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, pts1, pts2)  # Perform triangulation
    points_3d = points_4d[:3] / points_4d[3]  # Convert homogeneous coordinates to 3D coordinates
    return points_3d.T  # Transpose the result to match the original shape

def plot3d(coords):
    fig = plt.figure()
    plot_3d = fig.add_subplot(111, projection='3d')

    # Extracting x, y, z coordinates from the input list
    x = [coord[0] for coord in coords]
    y = [coord[1] for coord in coords]
    z = [coord[2] for coord in coords]

    # Plotting the points in 3D space
    plot_3d.scatter(x, y, z)

    # Adding labels and title
    plot_3d.set_xlabel('X Label')
    plot_3d.set_ylabel('Y Label')
    plot_3d.set_zlabel('Z Label')
    plot_3d.set_title('3D Scatter Plot')

    # Show the plot
    plt.show()
    plt.savefig('trajectory.png')


# Camera intrinsic parameters (K matrix)
K = np.array([[8.28013819e+02, 0.00000000e+00, 1.01014119e+03],
              [0.00000000e+00, 8.20546041e+02, 8.21208275e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# Rotation vectors for each camera
rvecs = [np.array([[-0.40848219], [-0.04476641], [-1.36582084]]),
         np.array([[-0.57560136], [0.06694504], [0.01289696]]),
         np.array([[-0.0139436], [1.02401999], [2.95480862]])]

# Translation vectors for each camera
tvecs = [np.array([[-2.64222845], [0.4595596], [1.91394887]]),
         np.array([[-1.20345239], [-2.50698575], [3.88573475]]),
         np.array([[1.15682178], [1.07354459], [1.30999783]])]

# Calculate projection matrices for each camera by combining intrinsic and extrinsic parameters
proj_matrix1 = K @ np.hstack((cv2.Rodrigues(rvecs[0])[0], tvecs[0]))
proj_matrix2 = K @ np.hstack((cv2.Rodrigues(rvecs[1])[0], tvecs[1]))
proj_matrix3 = K @ np.hstack((cv2.Rodrigues(rvecs[2])[0], tvecs[2]))

















# # Function to filter outliers based on expected ranges for the 3D points
# def filter_outliers(points_3d, x_range, y_range, z_range):
#     # Create a mask to filter points within the expected x, y, z ranges
#     mask = ((points_3d[:, 0] >= x_range[0]) & (points_3d[:, 0] <= x_range[1]) &
#             (points_3d[:, 1] >= y_range[0]) & (points_3d[:, 1] <= y_range[1]) &
#             (points_3d[:, 2] >= z_range[0]) & (points_3d[:, 2] <= z_range[1]))
#     return points_3d[mask]  # Return only the points that are within the expected ranges

# # Camera intrinsic parameters (K matrix)
# K = np.array([[8.28013819e+02, 0.00000000e+00, 1.01014119e+03],
#               [0.00000000e+00, 8.20546041e+02, 8.21208275e+02],
#               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# # Rotation vectors for each camera
# rvecs = [np.array([[-0.40848219], [-0.04476641], [-1.36582084]]),
#          np.array([[-0.57560136], [0.06694504], [0.01289696]]),
#          np.array([[-0.0139436], [1.02401999], [2.95480862]])]

# # Translation vectors for each camera
# tvecs = [np.array([[-2.64222845], [0.4595596], [1.91394887]]),
#          np.array([[-1.20345239], [-2.50698575], [3.88573475]]),
#          np.array([[1.15682178], [1.07354459], [1.30999783]])]

# # Calculate projection matrices for each camera by combining intrinsic and extrinsic parameters
# proj_matrix1 = K @ np.hstack((cv2.Rodrigues(rvecs[0])[0], tvecs[0]))
# proj_matrix2 = K @ np.hstack((cv2.Rodrigues(rvecs[1])[0], tvecs[1]))
# proj_matrix3 = K @ np.hstack((cv2.Rodrigues(rvecs[2])[0], tvecs[2]))

# # Triangulate 3D points for each bird using pairs of camera views
# points_3d_bird1_cam1_cam2 = triangulate_points(selected_points_cam1['bird_1'], selected_points_cam2['bird_1'], proj_matrix1, proj_matrix2)
# points_3d_bird1_cam2_cam3 = triangulate_points(selected_points_cam2['bird_1'], selected_points_cam3['bird_1'], proj_matrix2, proj_matrix3)
# points_3d_bird1_cam1_cam3 = triangulate_points(selected_points_cam1['bird_1'], selected_points_cam3['bird_1'], proj_matrix1, proj_matrix3)

# points_3d_bird2_cam1_cam2 = triangulate_points(selected_points_cam1['bird_2'], selected_points_cam2['bird_2'], proj_matrix1, proj_matrix2)
# points_3d_bird2_cam2_cam3 = triangulate_points(selected_points_cam2['bird_2'], selected_points_cam3['bird_2'], proj_matrix2, proj_matrix3)
# points_3d_bird2_cam1_cam3 = triangulate_points(selected_points_cam1['bird_2'], selected_points_cam3['bird_2'], proj_matrix1, proj_matrix3)

# # Define the expected physical dimensions of the aviary for outlier filtering
# x_range = (0, 2.44)
# y_range = (0, 4.572)
# z_range = (-0.5, 0.5)

# # Filter out 3D points that fall outside the expected range
# points_3d_bird1_cam1_cam2 = filter_outliers(points_3d_bird1_cam1_cam2, x_range, y_range, z_range)
# points_3d_bird1_cam2_cam3 = filter_outliers(points_3d_bird1_cam2_cam3, x_range, y_range, z_range)
# points_3d_bird1_cam1_cam3 = filter_outliers(points_3d_bird1_cam1_cam3, x_range, y_range, z_range)

# points_3d_bird2_cam1_cam2 = filter_outliers(points_3d_bird2_cam1_cam2, x_range, y_range, z_range)
# points_3d_bird2_cam2_cam3 = filter_outliers(points_3d_bird2_cam2_cam3, x_range, y_range, z_range)
# points_3d_bird2_cam1_cam3 = filter_outliers(points_3d_bird2_cam1_cam3, x_range, y_range, z_range)

# # Function to compute the average of 3D points
# def compute_average_point(points_3d):
#     return np.mean(points_3d, axis=0)

# # Compute average 3D points for each bird from each pair of camera views
# avg_point_bird1_cam1_cam2 = compute_average_point(points_3d_bird1_cam1_cam2)
# avg_point_bird1_cam2_cam3 = compute_average_point(points_3d_bird1_cam2_cam3)
# avg_point_bird1_cam1_cam3 = compute_average_point(points_3d_bird1_cam1_cam3)

# avg_point_bird2_cam1_cam2 = compute_average_point(points_3d_bird2_cam1_cam2)
# avg_point_bird2_cam2_cam3 = compute_average_point(points_3d_bird2_cam2_cam3)
# avg_point_bird2_cam1_cam3 = compute_average_point(points_3d_bird2_cam1_cam3)

# # Function to visualize the average 3D points from the perspective of each camera
# def visualize_avg_3d_points(avg_point_bird1, avg_point_bird2, rvecs, tvecs, filename_prefix):
#     fig = plt.figure(figsize=(18, 6))
    
#     # Plot from Camera 1 perspective
#     ax1 = fig.add_subplot(131, projection='3d')
#     ax1.scatter(avg_point_bird1[0], avg_point_bird1[1], avg_point_bird1[2], c='r', marker='x', s=100, label='Bird 1 Avg')
#     ax1.scatter(avg_point_bird2[0], avg_point_bird2[1], avg_point_bird2[2], c='b', marker='x', s=100, label='Bird 2 Avg')
#     ax1.view_init(elev=-np.rad2deg(rvecs[0][0][0]), azim=np.rad2deg(rvecs[0][2][0]))
#     ax1.set_xlabel('X (meters)')
#     ax1.set_ylabel('Y (meters)')
#     ax1.set_zlabel('Z (meters)')
#     ax1.set_title('Camera 1 Perspective')
#     ax1.legend()
    
#     # Plot from Camera 2 perspective
#     ax2 = fig.add_subplot(132, projection='3d')
#     ax2.scatter(avg_point_bird1[0], avg_point_bird1[1], avg_point_bird1[2], c='r', marker='x', s=100, label='Bird 1 Avg')
#     ax2.scatter(avg_point_bird2[0], avg_point_bird2[1], avg_point_bird2[2], c='b', marker='x', s=100, label='Bird 2 Avg')
#     ax2.view_init(elev=-np.rad2deg(rvecs[1][0][0]), azim=np.rad2deg(rvecs[1][2][0]))
#     ax2.set_xlabel('X (meters)')
#     ax2.set_ylabel('Y (meters)')
#     ax2.set_zlabel('Z (meters)')
#     ax2.set_title('Camera 2 Perspective')
#     ax2.legend()
    
#     # Plot from Camera 3 perspective
#     ax3 = fig.add_subplot(133, projection='3d')
#     ax3.scatter(avg_point_bird1[0], avg_point_bird1[1], avg_point_bird1[2], c='r', marker='x', s=100, label='Bird 1 Avg')
#     ax3.scatter(avg_point_bird2[0], avg_point_bird2[1], avg_point_bird2[2], c='b', marker='x', s=100, label='Bird 2 Avg')
#     ax3.view_init(elev=-np.rad2deg(rvecs[2][0][0]), azim=np.rad2deg(rvecs[2][2][0]))
#     ax3.set_xlabel('X (meters)')
#     ax3.set_ylabel('Y (meters)')
#     ax3.set_zlabel('Z (meters)')
#     ax3.set_title('Camera 3 Perspective')
#     ax3.legend()
    
#     fig.suptitle('Average 3D Points from Camera Perspectives')
#     plt.savefig(f'{filename_prefix}.png')  # Save the visualization as a PNG image
#     plt.show()

# # Visualize average 3D points from each camera perspective for both birds
# visualize_avg_3d_points(avg_point_bird1_cam1_cam2, avg_point_bird2_cam1_cam2, rvecs, tvecs, 'avg_3d_points_perspective_cam1_cam2')
# visualize_avg_3d_points(avg_point_bird1_cam2_cam3, avg_point_bird2_cam2_cam3, rvecs, tvecs, 'avg_3d_points_perspective_cam2_cam3')
# visualize_avg_3d_points(avg_point_bird1_cam1_cam3, avg_point_bird2_cam1_cam3, rvecs, tvecs, 'avg_3d_points_perspective_cam1_cam3')

# # Function to visualize the manually selected bird points in the images
# def visualize_birds_in_images(image_path, points, color):
#     image = cv2.imread(image_path)  # Load the image
#     for point in points:
#         cv2.circle(image, point, 5, color, -1)  # Draw circles at the selected points
#     return image

# # Visualize bird 1's points in images from each camera
# proj_image1_bird1 = visualize_birds_in_images(image_path1, selected_points_cam1['bird_1'], (0, 0, 255))  # Red for bird 1
# proj_image2_bird1 = visualize_birds_in_images(image_path2, selected_points_cam2['bird_1'], (0, 0, 255))
# proj_image3_bird1 = visualize_birds_in_images(image_path3, selected_points_cam3['bird_1'], (0, 0, 255))

# # Visualize bird 2's points in images from each camera
# proj_image1_bird2 = visualize_birds_in_images(image_path1, selected_points_cam1['bird_2'], (255, 0, 0))  # Blue for bird 2
# proj_image2_bird2 = visualize_birds_in_images(image_path2, selected_points_cam2['bird_2'], (255, 0, 0))
# proj_image3_bird2 = visualize_birds_in_images(image_path3, selected_points_cam3['bird_2'], (255, 0, 0))

# # Save and display the images with projected points for bird 1
# cv2.imwrite('proj_image_1_bird1.png', proj_image1_bird1)
# cv2.imwrite('proj_image_2_bird1.png', proj_image2_bird1)
# cv2.imwrite('proj_image_3_bird1.png', proj_image3_bird1)

# # Save and display the images with projected points for bird 2
# cv2.imwrite('proj_image_1_bird2.png', proj_image1_bird2)
# cv2.imwrite('proj_image_2_bird2.png', proj_image2_bird2)
# cv2.imwrite('proj_image_3_bird2.png', proj_image3_bird2)

# # Display images with manually selected points for bird 1 across the three cameras
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.imshow(cv2.cvtColor(proj_image1_bird1, cv2.COLOR_BGR2RGB))
# plt.title('Projected Bird 1 - Camera 1')
# plt.subplot(1, 3, 2)
# plt.imshow(cv2.cvtColor(proj_image2_bird1, cv2.COLOR_BGR2RGB))
# plt.title('Projected Bird 1 - Camera 2')
# plt.subplot(1, 3, 3)
# plt.imshow(cv2.cvtColor(proj_image3_bird1, cv2.COLOR_BGR2RGB))
# plt.title('Projected Bird 1 - Camera 3')
# plt.show()

# # Display images with manually selected points for bird 2 across the three cameras
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.imshow(cv2.cvtColor(proj_image1_bird2, cv2.COLOR_BGR2RGB))
# plt.title('Projected Bird 2 - Camera 1')
# plt.subplot(1, 3, 2)
# plt.imshow(cv2.cvtColor(proj_image2_bird2, cv2.COLOR_BGR2RGB))
# plt.title('Projected Bird 2 - Camera 2')
# plt.subplot(1, 3, 3)
# plt.imshow(cv2.cvtColor(proj_image3_bird2, cv2.COLOR_BGR2RGB))
# plt.title('Projected Bird 2 - Camera 3')
# plt.show()