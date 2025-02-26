# 3D_Multi_Bird_Tracking

This is the official repository representing Keon Moradi, Ethan Haque, Jasmeen Kaur, Eli Bridge, Alexandra Bentz, and Golnaz Habibi in their work _Context-Aware Outlier Rejection for Robust Multi-View 3D Tracking of Similar Small Birds in An Outdoor Aviary_ accepted into the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025. 

### Clone Git Repo
```shell
git clone https://github.com/airou-lab/3D_Multi_Bird_Tracking.git
```

Upon cloning the package, the only script neccessary to run is the ```clean.py``` script within the ```pipeline``` folder. Initially, you will be prompted to select the location of various predefined landmarks within your enclosure environment by using your cursor to click on their location in the first frame of each relevant camera view. These can be changed to suit different environments by adding, removing, or redefining landmarks. These will be used to define the Voronoi graph the lays on top of all future image frames from the corresponding camera view. Moreover, make sure to define your desired camera pairings at the top of the script based on compatability in views between cameras. 

Ensure that you have placed the paths to your video frames in the array of file paths at the top of script and that you have obtained your detections in a csv format that includes frame number and bounded box coordinates.


### Fine Tuning & Annotating the Model

1). Make a directory to store all the videos that you are going to run.
```bash
mkdir videos
```
2). Put all the videos in the newly made directory

3). Run Detection with YoloV5 with pre-trained model.
```bash
cd yolov5
pip install -r requirements.txt
```



Run the pretrained model:
Note: Change the name video1.mp4 to the actual video that you want to test
```bash
python detect.py --source ../videos/video1.mp4 --weights yolov5s.pt --conf 0.25 --save-txt --save-conf
```

You should be able to see results in the yolov5 folder -> runs -> detect -> exp

 

4). To run the fine-tuned model run the following command.

Note: Change the name video1.mp4 to the actual video that you want to test
```bash
python detect.py --weights runs/train/exp7/weights/best.pt --source ../videos/video1.mp4 --conf 0.25 --view-img --save-csv
```

You should be able to see results in the yolov5 folder -> runs -> detect -> exp2

or it will print out where the video and results are save within the terminal output





### How to Fine tune the detection model.


1). Save Frames from Video at Regular Intervals

Essentially, run the process_videos.py code

Note: 1). Input and Output: The script assumes that all videos are stored in a folder named videos and will save the extracted frames to dataset/images.
2). Video Processing: It lists all .mp4 files in the videos folder, processes each one, saves frames at the specified interval, and names the frames according to their source video and frame number.
3). Frame Rate and Interval: Frames are saved based on the frame_rate variable, which you can adjust according to how frequently you want to capture frames.


2). Install Docker: If Docker is not already installed on your computer.

3). Install CVAT:
Open a terminal and clone the CVAT repository:

```bash
git clone https://github.com/openvinotoolkit/cvat.git
cd cvat
```

Set up CVAT using Docker:
```bash
docker-compose up -d
```

This command will download and start all the necessary Docker containers. Once the process is complete, CVAT will be accessible via http://localhost:8080.


4). Create an Account and Log In

Open a web browser and go to http://localhost:8080. Next, Click on 'Sign Up' to create a new account. Finally, After signing up, log in with your new credentials.

5).Create a New Project:
Go to the "Projects" tab, Click on "Create new project." Fill in the project name and description. For example: Name: "Bird Detection - New Batch"

6). Next Add Labels:
Click on the “Add label” button to create a new label for your annotations. Since you're focusing on birds, you should add a label named "bird". Shape: Select "Rectangle" from the dropdown menu. No need to set up a skeleton or use a model for this simple bounding box annotation.

7). Advanced Configuration: You can generally leave the "Advanced configuration" section as it is unless you have specific settings or plugins you wish to use. Source and Target Storage: Since you’re likely using local files and saving annotations locally, ensure both source and target storage are set to “Local”. Make sure everything looks correct, and then click "Submit & Open" or "Submit & Continue."

8). Create a new task for annotating your images

Name: Change the task name to something more general like "Bird Detection Across Videos date 8_7_2024" if you intend to upload images from all five videos into this single task.

Project: It should automatically select the project you just created.

Subset: You can leave this blank unless you're planning to specifically categorize your tasks into subsets (e.g., training, validation).

Select Files: When selecting files to upload, you can choose frames from all five videos if they're stored in the same directory.

Click "Submit" to create the task with all the selected images.

This process will consolidate your annotation work, allowing you to handle all video frames within a single task, making it easier to export and manage the dataset later.

9). Annotating

On the left side of the screen, you’ll see a toolbar with various tools. For annotating birds, you will primarily use the rectangle tool (the icon that looks like a rectangle). Select the Rectangle Tool: Click on the rectangle icon in the toolbar. Shape: Use this for drawing individual rectangles (or other shapes, depending on what you select) around objects. Each shape is independent of others, and you manually place them around each bird you need to annotate in the image. Draw a Rectangle: Click and drag on the image to draw a rectangle around each bird you see in the frame. After drawing the rectangle, ensure the label "bird" is assigned to it. If it's not automatically selected, you can choose it from a dropdown menu that appears when you select the rectangle.

10). Navigate through the Images

Use the playback controls (the arrow buttons near the top of the screen) to move through the frames you’ve uploaded. You need to annotate birds in each frame where they are visible.

11). Save Your Annotations

Regularly save your progress by clicking the "Save" button at the top of the screen. This is crucial to ensure you don't lose your work.

12). Do one final save and then export Annotations.

There should be a "Save" button in the upper menu bar of the annotation interface. 

Navigate to the Task Dashboard:
Exit the annotation interface by going back to the main dashboard of CVAT where your tasks are listed. This is typically done by clicking on the "Menu" and selecting “Tasks”.

Find Your Task:
Locate the task you were working on from the list of tasks (e.g., "Bird Detection Across Videos").

Export Annotations:
Click on your task to open the task details. Look for an option or button labeled "Export" or "Export Task".

Export Format:
"YOLO 1.1", which is appropriate for training YOLO models. This will format the annotation files according to the YOLO specifications, with each line in a .txt file corresponding to a bounding box and containing class ID, x-center, y-center, width, and height, all normalized to the image size. If you toggle "Save images" on, it will include the images along with the annotation files in the zip file you download. This is useful if you want to have a complete set of images and corresponding annotations together, especially if you made any modifications or if it’s more convenient for your training setup. If you already have all the original images organized and just need the annotation files, you can leave this toggled off.

Custom Name:
If you want to give a specific name to your dataset file, you can enter it in the "Custom name for a dataset" field. If left blank, CVAT will generate a default name based on the task name and the export format.


Use Default Settings:
This is usually fine for most exports, ensuring that the annotations are exported with the required configurations for the YOLO format.

Finalize Export:
Click "OK" to start the export process. Depending on the size of the dataset and your network connection, this might take a moment.

After you click "OK", the dataset will be prepared and downloaded to your computer as a .zip file. Once downloaded, you can unzip it to verify the contents, ensuring that the image files (if you chose to include them) and the annotation .txt files are all correct and correspond to each other.


13. Organize Your Dataset

To ensure smooth operation during model training, your dataset should be well-organized, typically as follows:

Images Folder: All your images should be in one folder. It looks like you already have an images folder ready.

Labels Folder: The annotation .txt files should be in a separate folder but mirror the structure of the images folder. You can place the annotation files from the extracted folder into a labels folder at the same directory level as your images folder.


```markdown
dataset/
│
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── ...
│
└── labels/
    ├── image1.txt
    ├── image2.txt
    ├── ...


```


Validate File Correspondence:
Ensure Correspondence: Check that every image file in the images folder has a corresponding annotation file in the labels folder with the exact same filename (except for the file extension).


14). Install necessary tools and Libraries:

You'll need Python installed on your machine. If not already installed, download it from the official Python website.
Install PyTorch. If you are using a Mac without a dedicated GPU, install the CPU version of PyTorch:

```bash
pip install torch torchvision torchaudio
```

Install other required libraries, including YOLOv5 dependencies:
```bash
pip install matplotlib seaborn numpy pandas
```

If not done so already Download YOLOv5:

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
```

Install the reaming dependencies required by YOLOv5:
```bash
pip install -r requirements.txt
```


15). Prepare Your Dataset for Training

Create a Data Configuration File: Create a YAML file that specifies the paths to your datasets and the number of classes. For example, create a file named "birds_dataset_1.yaml" or change '_1' to whatever version of fine tuning you are on.

In the yaml file:
```yaml
train: ../dataset/images  # path to training images
val: ../dataset/images    # path to validation images (can be the same for now)

nc: 1  # number of classes
names: ['bird']  # list of class names
```

16). Organize Your Dataset and train the model:
Ensure your images and corresponding labels are placed as specified in the YAML file. It's good to divide your dataset into a training set and a validation set, but for simplicity, you can initially use the same set for both.


if this is your first time fine-tuning the yolo model use this command:
```bash
python train.py --img 640 --batch 16 --epochs 100 --data birds_dataset.yaml --weights yolov5s.pt --workers 2
```

change the "birds_dataset.yaml" to your yaml file name, and note that yolov5s.pt is the base model.

But if this is you are further Fine-tuning a previous Fine-tuned model use this command:

```bash
python train.py --img 640 --batch 16 --epochs 100 --data data.yaml --weights runs/train/exp7/best.pt --cache --workers 2
```

Make sure to change the --weights path --weights runs/train/exp7/best.pt to where your current fine-tuned model is.

The --cache option in the YOLOv5 training command is used to speed up the training process by caching images into memory. This means that the images will be loaded into RAM during the first epoch and then reused for subsequent epochs. This can significantly reduce the disk I/O time and improve training speed, especially if you have a large number of images and sufficient RAM.

The --workers option specifies the number of DataLoader workers. DataLoader workers are used to load data in parallel, which can speed up data loading, especially for larger datasets. By default, YOLOv5 will use multiple workers, but you can increase or decrease the number based on your system's capabilities.


Note when I tried running these steps I kept get errors on MacOS due to SSL certificate verification on macOS. I had to use ChatGPT to fix the error, but once you do that if you run into these issues then you will be able to train the model
### RUnning the outlier rejection pipeline


```bash
python pipeline/go.py 
```
