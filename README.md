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

### Usage
* To visualize rendering in blender, open a scripting window in Blender, load rope-blender.py, and run
* Select the spline in 'Edit' mode, press 'G,' and mouseover any of the spline nodes to move the rope
* To generate data in background, run `blender -b -P rope-blender.py`
