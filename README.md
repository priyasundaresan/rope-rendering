# rope-rendering

## rope-blender.py
### Dependencies
* Blender (beta version 2.8)
* OpenCV
* PyYAML

### Description
* This repo hosts rendering scripts for the dense descriptors for deformable objects project. 
  * `rope-blender.py`: renders images of deformable rope into directory `rope-rendering/images` and dumps a YAML file with vertex pixel info into the same folder
  * `vis.py`: loads a set of images and a YAML file from `rope-rending/images` and highlights all the pixels in the YAML file on each image. Dumps annotated images into `rope-rendering/annotated`
  * `mask.py`: renders a visible mask (255's and 0's) and non-visible mask (1's and 0's) of each image and dumps them to `rope-rendering/image_masks.`

### Setup
* After downloading Blender version 2.8, do the following steps:
* Add the following line to your .bashrc: 
  * `alias blender="/Users/priyasundaresan/Downloads/blender-2.80.0-git20190620.d30f72dfd8ac-x86_64/blender.app/Contents/MacOS/blender"` replacing the path to blender.app with your downloaded version
* `cd` into the following directory: `/Users/priyasundaresan/Downloads/blender-2.80.0-git20190620.d30f72dfd8ac-x86_64/blender.app/Contents/Resources/2.80/python/bin`
  * This is Blender's Python installation (different from system Python installation). Install PyYAML:
  * `curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`
  * `./python3.7m get-pip.py`
  * `./python3.7m pip3 install PyYAML`

### Rendering Usage
* Off-screen rendering: run `blender -b -P rope-blender.py`
* On-screen rendering: run `blender -P rope-blender.py`
* For debugging purposes, you can also open a scripting window in Blender, copy or load `rope-blender.py` and hit `Run Script`
  * To manually deform the road, select the spline in `'EDIT'` mode, press `'G'`, and grab any of the spline nodes with a mouse to move the rope. You can select groups of nodes at a time by pressing `'Ctrl'` while grabbing nodes.

