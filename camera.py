import bpy
import bpy_extras
from mathutils import *

scene = bpy.context.scene
obj = bpy.context.object
co = Vector((0, 1, 0))

co_2d = bpy_extras.object_utils.world_to_camera_view(scene, obj, co)
print("2D Coords:", co_2d)

# If you want pixel coords
render_scale = scene.render.resolution_percentage / 100
render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
        )
print("Pixel Coords:", (
      round(co_2d.x * render_size[0]),
      round(co_2d.y * render_size[1]),
      ))
cam_loc = bpy.data.objects["Camera"].location
cam_rot = bpy.data.objects["Camera"].rotation_euler
print(cam_loc, cam_rot)
