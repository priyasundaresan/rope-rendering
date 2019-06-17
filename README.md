# rope-rendering

## rope.py
### Dependencies
* Panda3D (pip install panda3d==1.10.3)
* A handful of Python libraries (OpenCV, Numpy, Scipy (version 1.3.0))

### Usage
* `$ python3 rope.py` by default renders a dynamic rope
* Add flat `--static` to render a static rope instead, and add flag `--random` to randomly change the camera pose (for synthetically generating data)

## rope-blender.py
### Dependencies
* Blender

### Usage
* Open a scripting window in Blender, load rope-blender.py, and run
* Select the spline in 'Edit' mode, press 'G,' and mouseover any of the spline nodes to move the rope
