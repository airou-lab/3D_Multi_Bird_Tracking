# 3D_Multi_Bird_Tracking

### Clone Git Repo
```shell
git clone https://github.com/airou-lab/3D_Multi_Bird_Tracking.git
```

Upon cloning the package, the only script neccessary to run is the ```clean.py``` script within the ```pipeline``` folder. Initially, you will be prompted to select the location of various predefined landmarks within your enclosure environment by using your cursor to click on their location in the first frame of each relevant camera view. These can be changed to suit different environments by adding, removing, or redefining landmarks. These will be used to define the Voronoi graph the lays on top of all future image frames from the corresponding camera view. Moreover, make sure to define your desired camera pairings at the top of the script based on compatability in views between cameras. 

Ensure that you have placed the paths to your video frames in the array of file paths at the top of script and that you have obtained your detections in a csv format that includes frame number and bounded box coordinates.
