# rope-rendering

## rope-blender.py
### Dependencies
* Blender (beta version 2.8)
* OpenCV
* PyYAML

### Setup
* After downloading Blender version 2.8, do the following steps:
* Add the following line to your .bashrc: 
  * `alias blender="/Users/priyasundaresan/Downloads/blender-2.80.0-git20190620.d30f72dfd8ac-x86_64/blender.app/Contents/MacOS/blender"` replacing the path to blender.app with your downloaded version
* `cd` into the following directory: `/Users/priyasundaresan/Downloads/blender-2.80.0-git20190620.d30f72dfd8ac-x86_64/blender.app/Contents/Resources/2.80/python/bin`
 * This is Blender's Python installation, which is different from your OS's Python installation. One of the dependencies for exporting Bezier knot/vertex info about each picture is PyYAML, which we will install here as follows:
 * `curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`
 * `./python3.7m get-pip.py`
 * `./python3.7m pip3 install PyYAML`

### Usage
* To visualize rendering in blender, open a scripting window in Blender, load rope-blender.py, and run
* Select the spline in 'Edit' mode, press 'G,' and mouseover any of the spline nodes to move the rope
* To generate data in background, run `blender -b -P rope-blender.py`
