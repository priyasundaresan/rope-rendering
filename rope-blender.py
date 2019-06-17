import bpy
from math import *
from mathutils import *


# Delete any objects already in the scene
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Make the rope (based off this tutorial: https://youtu.be/xYhIoiOnPj4)
context = bpy.context
bpy.ops.mesh.primitive_circle_add(location=(0, 0, 0))
bpy.ops.transform.resize(value=(0.1, 0.1, 0.1))
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.transform.rotate(value= pi / 2, orient_axis='X')
bpy.ops.transform.translate(value=(0.1, 0, 0))
bpy.ops.object.mode_set(mode='OBJECT')
circle = context.active_object
circle.name = "Circle0"
for i in range(1, 4):
    bpy.ops.object.duplicate_move(OBJECT_OT_duplicate=None, TRANSFORM_OT_translate=None)
    ob = bpy.context.active_object
    ob.name = "Circle%d" % i
    ob.rotation_euler = (0, 0, i * (pi / 2))
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.join()
rope = context.active_object
rope.name = "Rope"
bpy.ops.object.modifier_add(type='SCREW')
rope.modifiers["Screw"].screw_offset = 10
rope.modifiers["Screw"].iterations = 10
bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Screw")

# Add a cone and a sphere at either end of the rope for asymmetry
bpy.ops.mesh.primitive_cone_add(location=(0, 0.32, 0))
bpy.ops.transform.resize(value=(0.5, 0.5, 0.5))
bpy.ops.transform.rotate(value= pi / 2, orient_axis='X')
bpy.ops.mesh.primitive_ico_sphere_add(radius=0.5, location=(0, 0, 10))
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.join()
rope_asymm = context.active_object
rope_asymm.name="Rope-Asymmetric"

# Create bezier curve and attach rope to it
bpy.ops.curve.primitive_bezier_curve_add(location=(0, 0, 0))
bpy.ops.transform.resize(value=(7, 7, 7))
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.curve.subdivide(number_cuts=7)
bpy.ops.object.mode_set(mode='OBJECT')
bezier = context.active_object
bezier.name = "Bezier"
bpy.data.objects['Bezier'].select_set(False)
bpy.data.objects['Rope-Asymmetric'].select_set(True)
context.view_layer.objects.active = rope_asymm
bpy.ops.object.modifier_add(type='CURVE')
rope_asymm.modifiers["Curve"].deform_axis = 'POS_Z'
rope_asymm.modifiers["Curve"].object = bpy.data.objects["Bezier"]
