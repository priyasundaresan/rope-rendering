import bpy
from math import *
from mathutils import *
import pprint
import numpy as np
import random


class RopeRenderer:
    def __init__(self, rope_radius=0.1, rope_screw_offset=10, rope_iterations=10, bezier_scale=7, bezier_subdivisions=10):
        self.rope_radius = rope_radius # thickness of rope
        self.rope_iterations = rope_iterations # how many "screws" are stacked lengthwise to create the rope
        self.rope_screw_offset = rope_screw_offset # how tightly wound the "screw" is
        self.bezier_scale = bezier_scale # length of bezier curve
        self.bezier_subdivisions = bezier_subdivisions # the number of splits in the bezier curve (ctrl points - 1)
        self.origin = (0, 0, 0)
        self.rope = None
        self.asymm_rope = None
        self.bezier = None
        self.rope_name = "Rope"
        self.asymm_rope_name = "Rope-Asymmetric"
        self.bezier_name = "Bezier"

    def clear(self):
        # Delete any existing objects in the scene
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def make_rigid_rope(self):
        # Join 4 circles and "twist" them to create realistic rope (based off this tutorial: https://youtu.be/xYhIoiOnPj4)
        bpy.ops.mesh.primitive_circle_add(location=self.origin)
        bpy.ops.transform.resize(value=(self.rope_radius, self.rope_radius, self.rope_radius))
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.transform.rotate(value= pi / 2, orient_axis='X')
        bpy.ops.transform.translate(value=(self.rope_radius, 0, 0))
        bpy.ops.object.mode_set(mode='OBJECT')
        for i in range(1, 4):
            bpy.ops.object.duplicate_move(OBJECT_OT_duplicate=None, TRANSFORM_OT_translate=None)
            ob = bpy.context.active_object
            ob.rotation_euler = (0, 0, i * (pi / 2))
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.join()
        self.rope = bpy.context.active_object
        self.rope.name = self.rope_name
        bpy.ops.object.modifier_add(type='SCREW')
        self.rope.modifiers["Screw"].screw_offset = self.rope_screw_offset
        self.rope.modifiers["Screw"].iterations = self.rope_iterations
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Screw")

    def add_rope_asymmetry(self):
        # Add a cone and icosphere at either end, to break symmetry
        bpy.ops.mesh.primitive_cone_add(location=(self.origin[0], self.origin[1] + 0.32, self.origin[2]))
        bpy.ops.transform.resize(value=(0.5, 0.5, 0.5))
        bpy.ops.transform.rotate(value= pi / 2, orient_axis='X')
        bpy.ops.mesh.primitive_ico_sphere_add(radius=0.5, location=(self.origin[0], self.origin[1], self.origin[2] + self.rope_radius*self.rope_screw_offset*self.rope_iterations))
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.join()
        self.rope_asymm = bpy.context.active_object
        self.rope_asymm.name= self.asymm_rope_name

    def make_bezier(self):
        # Create bezier curve, attach rope to it
        bpy.ops.curve.primitive_bezier_curve_add(location=self.origin)
        bpy.ops.transform.resize(value=(self.bezier_scale, self.bezier_scale, self.bezier_scale))
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.curve.subdivide(number_cuts=self.bezier_subdivisions)
        bpy.ops.transform.resize(value=(1, 0, 1))
        bpy.ops.object.mode_set(mode='OBJECT')
        self.bezier = bpy.context.active_object
        self.bezier.name = self.bezier_name
        self.bezier.select_set(False)
        self.rope_asymm.select_set(True)
        bpy.context.view_layer.objects.active = self.rope_asymm
        bpy.ops.object.modifier_add(type='CURVE')
        self.rope_asymm.modifiers["Curve"].deform_axis = 'POS_Z'
        self.rope_asymm.modifiers["Curve"].object = self.bezier
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')

    def randomize_nodes(self, num):
    	bez_points = self.bezier.data.splines[0].bezier_points
    	nodes = np.random.choice(bez_points, min(num, len(bez_points)), replace=False)
    	for node in nodes:
	    	node.co.x += random.uniform(-0.2, 0.2)
	    	node.co.y += random.uniform(-0.2, 0.2)
	    	node.co.z += random.uniform(-0.2, 0.2)
        # bez_points = self.bezier.data.splines[0].bezier_points
        # sz = len(bez_points)
        # for i in range(0, sz):
        #     # Set every sixth coordinate's z to 0.5.
        #     bez_points[i].co.z += random.uniform(0, 0.2)

if __name__ == '__main__':
    renderer = RopeRenderer(rope_radius=0.1, rope_screw_offset=10, rope_iterations=10, bezier_scale=6, bezier_subdivisions=7)
    renderer.clear()
    renderer.make_rigid_rope()
    renderer.add_rope_asymmetry()
    renderer.make_bezier()
    renderer.randomize_nodes(3)
