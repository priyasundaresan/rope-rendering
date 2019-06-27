import bpy, bpy_extras
from math import *
from mathutils import *
import pprint
import random
import yaml
import numpy as np
# import cv2

class RopeRenderer:
    def __init__(self, rope_radius=0.1, rope_screw_offset=10, rope_iterations=10, bezier_scale=7, bezier_subdivisions=10, save=False):
        # Hyperparams for the rope
        self.save = save
        self.rope_radius = rope_radius # thickness of rope
        self.rope_iterations = rope_iterations # how many "screws" are stacked lengthwise to create the rope
        self.rope_screw_offset = rope_screw_offset # how tightly wound the "screw" is
        self.bezier_scale = bezier_scale # length of bezier curve
        self.bezier_subdivisions = bezier_subdivisions # the number of splits in the bezier curve (ctrl points - 2)
        self.origin = (0, 0, 0)
        # Make objects
        self.rope = None
        self.asymm_rope = None
        self.bezier = None
        self.bezier_points = None
        self.camera = None
        # Name objects
        self.rope_name = "Rope"
        self.asymm_rope_name = "Rope-Asymmetric"
        self.bezier_name = "Bezier"
        self.camera_name = "Camera"
        # Render stuff
        self.knots_info = {}
        self.i = 0
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        prefs.compute_device = 'CUDA_0'

    def clear(self):
        # Delete any existing objects in the scene, place a camera randomly
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        bpy.ops.object.camera_add(location=[random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
        self.camera = bpy.context.active_object
        self.camera.name = self.camera_name
        self.camera.rotation_euler = (random.uniform(-pi/4, pi/4), random.uniform(-pi/4, pi/4), random.uniform(-pi, pi))


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
        bpy.ops.transform.resize(value=(0.25, 0.25, 0.25))
        bpy.ops.transform.rotate(value= pi / 2, orient_axis='X')
        # bpy.ops.mesh.primitive_ico_sphere_add(radius=0.25, location=(self.origin[0], self.origin[1], self.origin[2] + self.rope_radius*self.rope_screw_offset*self.rope_iterations))
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
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.curve.select_all(action='SELECT')
        bpy.ops.curve.handle_type_set(type='AUTOMATIC')
        bpy.ops.object.mode_set(mode='OBJECT')
        self.bezier = bpy.context.active_object
        self.bezier_points = self.bezier.data.splines[0].bezier_points
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

    def randomize_nodes(self, num, offset_min, offset_max, buffer, alpha):
        # Random select num knots, pull them in a random direction (constrained by offset_min, offset_max)
        knots_idxs = np.random.choice(range(buffer - 1, len(self.bezier_points) - buffer, buffer*2), min(num, len(self.bezier_points)), replace=False)

        for idx in knots_idxs:
            knot = self.bezier_points[idx]
            offset_y = random.uniform(offset_min, offset_max)
            offset_x = random.uniform(offset_min/2, offset_max/2)
            if random.uniform(0, 1) < 0.5:
                offset_y *= -1
            if random.uniform(0, 1) < 0.5:
                offset_x *= -1
            knot.co.y += offset_y
            knot.co.x += offset_x
            if idx + buffer > len(self.bezier_points):
                r = range(-1, -buffer - 1, -1)
            else:
                r = range(1, buffer + 1, 1)
            for b in r:
                idx_a = idx + b
                knot_a = self.bezier_points[idx_a]
                knot_a.co.y += alpha*offset_y/b
                knot_a.co.x += alpha*offset_x/b


    def reposition_camera(self):
        # Orient camera towards the rope
        bpy.context.scene.camera = self.camera
        bpy.ops.view3d.camera_to_view_selected()

    def in_bounds(self, pixel):
        return 0 <= pixel[0] <= 640 and 0 <= pixel[1] <= 480

    def render_single_scene(self):
		# Produce a single image of the current scene, save the bezier knot pixel coordinates
        scene = bpy.context.scene
        # knots = [self.bezier.matrix_world @ knot.co for knot in self.bezier_points]
        pixels = []
        scene.render.resolution_percentage = 100
        render_scale = scene.render.resolution_percentage / 100
        scene.render.resolution_x = 640
        scene.render.resolution_y = 480
        render_size = (
                int(scene.render.resolution_x * render_scale),
                int(scene.render.resolution_y * render_scale),
                )
        for j in range(len(knots)):
            knot_coord = knots[j]
            co_2d = bpy_extras.object_utils.world_to_camera_view(scene, self.camera, knot_coord)
            print("2D Coords:", co_2d)
            # If you want pixel coords
            render_scale = scene.render.resolution_percentage / 100
            scene.render.resolution_x = 640
            scene.render.resolution_y = 480
            render_size = (
                    int(scene.render.resolution_x * render_scale),
                    int(scene.render.resolution_y * render_scale),
                    )
            pixels.append([round(co_2d.x * render_size[0]), round(render_size[1] - co_2d.y * render_size[1])])
        bpy.context.scene.world.color = (1, 1, 1)
        bpy.context.scene.render.display_mode
        bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
        bpy.context.scene.display_settings.display_device = 'None'
        bpy.context.scene.sequencer_colorspace_settings.name = 'XYZ'
        if self.save and all([self.in_bounds(p) for p in pixels]):
            color_filename = "{0:06d}_rgb.png".format(self.i)
            self.knots_info[self.i] = pixels
            bpy.context.scene.render.image_settings.file_format='PNG'
            bpy.context.scene.render.filepath = "/Users/priyasundaresan/Desktop/rope-rendering/images/{}".format(color_filename)
            bpy.ops.render.render(use_viewport = True, write_still=True)
            self.i += 1

    def run(self, num_images):
        for i in range(num_images):
            self.clear()
            self.make_rigid_rope()
            self.add_rope_asymmetry()
            self.make_bezier()
            self.randomize_nodes(2, 0.2, 0.3, 4, 1)
            self.reposition_camera()
            self.render_single_scene()
        pprint.pprint(self.knots_info)
        with open("/Users/priyasundaresan/Desktop/rope-rendering/images/knots_info.yaml", 'w') as outfile:
            yaml.dump(self.knots_info, outfile, default_flow_style=False)

if __name__ == '__main__':
    renderer = RopeRenderer(rope_radius=0.05, rope_screw_offset=10, rope_iterations=20, bezier_scale=3.4, bezier_subdivisions=48, save=False)
    renderer.run(1)
