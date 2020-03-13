import bpy, bpy_extras
import sys
import copy
from math import *
import pprint
from mathutils import *
import mathutils
import random
import json
import numpy as np
import os
import time
from sklearn.neighbors import NearestNeighbors
import argparse

class RopeRenderer:
    def __init__(self, asymmetric=False, rope_radius=None, sphere_radius=None, rope_iterations=None, rope_screw_offset=None, bezier_scale=3.7, bezier_knots=12, save_depth=True, save_rgb=False, num_annotations=20, num_images=10, nonplanar=True, render_width=640, render_height=480, sequence=False, episode_length=1):
        self.save_rgb = save_rgb # whether to save_rgb images or not
        self.num_images = num_images
        self.sequence = sequence
        self.episode_length = episode_length
        self.render_width = render_width
        self.render_height = render_height
        self.num_annotations = num_annotations
        self.save_depth = save_depth

        self.asymmetric = asymmetric
        self.rope_radius = rope_radius
        self.rope_screw_offset = rope_screw_offset
        self.sphere_radius = sphere_radius
        self.rope_iterations = rope_iterations
        self.nonplanar = nonplanar
        self.bezier_scale = bezier_scale
        self.bezier_subdivisions = bezier_knots - 2 # the number of splits in the bezier curve (ctrl points - 2)
        self.origin = (0, 0, 0)
        # Make objects
        self.rope = None
        self.rope_asymm = None
        self.bezier = None
        self.straight_bezier_coords = None
        self.bezier_points = None # list of vertices in bezier curve
        self.camera = None
        # Name objects
        self.rope_name = "Rope"
        self.rope_asymm_name = "Rope-Asymmetric"
        self.bezier_name = "Bezier"
        self.camera_name = "Camera"
        # Dictionary to store pixel vals of knots (vertices)
        self.knots_info = {}
        self.i = 0

    def clear(self):
        """
        Deletes any objects or meshes in the scene
        """
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)
        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)
        for block in bpy.data.textures:
            if block.users == 0:
                bpy.data.textures.remove(block)
        for block in bpy.data.images:
            if block.users == 0:
                bpy.data.images.remove(block)
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def add_camera(self, fixed=True):
        '''
        Place a camera randomly, fixed means no rotations about z axis (planar camera changes only)
        '''
        if fixed:
            bpy.ops.object.camera_add(location=[0, 0, 10])
            self.camera = bpy.context.active_object
            self.camera.rotation_euler = (0, 0, random.uniform(0, pi/2)) # fixed z, rotate only about x/y axis slightly
        else:
            bpy.ops.object.camera_add(location=[random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
            self.camera = bpy.context.active_object
            self.camera.rotation_euler = (random.uniform(-pi/4, pi/4), random.uniform(-pi/4, pi/4), random.uniform(-pi, pi))
        self.camera.name = self.camera_name
        bpy.ops.object.light_add(type='SUN', radius=1, location=(0, 0, 0))

    def make_rigid_rope(self):
        bpy.ops.mesh.primitive_circle_add(location=self.origin)
        radius = np.random.uniform(0.035, 0.037) if self.rope_radius is None else self.rope_radius
        bpy.ops.transform.resize(value=(radius, radius, radius))
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.transform.rotate(value= pi / 2, orient_axis='X')
        bpy.ops.transform.translate(value=(radius, 0, 0))
        bpy.ops.object.mode_set(mode='OBJECT')
        for i in range(1, 4):
            bpy.ops.object.duplicate_move(OBJECT_OT_duplicate=None, TRANSFORM_OT_translate=None)
            ob = bpy.context.active_object
            ob.rotation_euler = (0, 0, i * (pi / 2))
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.join()
        self.rope = bpy.context.active_object
        
        self.rope_asymm = self.rope

        self.rope.name = self.rope_name
        bpy.ops.object.modifier_add(type='SCREW')
        screw_offset = np.random.uniform(12.5, 13) if self.rope_screw_offset is None else self.rope_screw_offset
        self.rope.modifiers["Screw"].screw_offset = screw_offset 
        rope_iterations = 18.8 if self.rope_iterations is None else self.rope_iterations
        self.rope.modifiers["Screw"].iterations = rope_iterations
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Screw")

    def add_rope_asymmetry(self):
        '''
        Add sphere, to break symmetry of the rope
        '''
        bpy.ops.mesh.primitive_uv_sphere_add(location=(self.origin[0], self.origin[1], self.origin[2]))
        sphere_radius = np.random.uniform(0.27, 0.27) if self.sphere_radius is None else self.sphere_radius
        bpy.ops.transform.resize(value=(sphere_radius, sphere_radius, sphere_radius))
        bpy.ops.transform.rotate(value= pi / 2, orient_axis='X')
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.join()
        self.rope_asymm = bpy.context.active_object
        self.rope_asymm.name= self.rope_asymm_name

    def make_bezier(self):
        '''
        Create bezier curve
        '''
        bpy.ops.curve.primitive_bezier_curve_add(location=self.origin)
        bezier_scale = np.random.uniform(2.85, 3.02) if self.bezier_scale is None else self.bezier_scale
        bpy.ops.transform.resize(value=(bezier_scale, bezier_scale, bezier_scale))
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.curve.subdivide(number_cuts=self.bezier_subdivisions)
        bpy.ops.transform.resize(value=(1, 0, 1))
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.curve.select_all(action='SELECT')
        bpy.ops.curve.handle_type_set(type='AUTOMATIC')
        bpy.ops.object.mode_set(mode='OBJECT')
        self.bezier = bpy.context.active_object
        self.bezier_points = self.bezier.data.splines[0].bezier_points
        self.straight_bezier_coords = [Vector(tuple(i.co)) for i in self.bezier_points]
        self.bezier.name = self.bezier_name
        self.bezier.select_set(False)
        self.rope_asymm.select_set(True)
        bpy.context.view_layer.objects.active = self.rope_asymm
        # Add bezier curve as deform modifier to the rope
        bpy.ops.object.modifier_add(type='CURVE')
        self.rope_asymm.modifiers["Curve"].deform_axis = 'POS_Z'
        self.rope_asymm.modifiers["Curve"].object = self.bezier
        self.rope_asymm.modifiers["Curve"].show_in_editmode = True
        self.rope_asymm.modifiers["Curve"].show_on_cage = True
        bpy.ops.object.mode_set(mode='EDIT')
        # Adding a curve can mess up surface normals of rope, re-point them outwards
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
    
    def straighten_rope(self):
        #self.rope_asymm.modifiers["Curve"].object = self.straight_bezier
        for i, p in enumerate(self.bezier_points):
            p.co = self.straight_bezier_coords[i]

    def make_simple_loop(self, x_offset_min, x_offset_max, y_offset_min, y_offset_max, p0=None):
        # Make simple cubic loop (4 control points)  + 2 more control points to pull slack through the loop
        #    2_______1
        #     \  4__/ 
        #      \ | /\
        #       \5/__\____________
        #       / \   | 
        #______0   3__|
        if p0 == None:
            p0 = np.random.choice(range(len(self.bezier_points) - 5))
        x_shift_1 = random.uniform(x_offset_min, x_offset_max)
        y_shift_1 = random.uniform(y_offset_min, y_offset_max)
        x_shift_2 = random.uniform(x_offset_min, x_offset_max)
        y_shift_2 = random.uniform(y_offset_min, y_offset_max)
        self.bezier_points[p0 + 1].co.y += y_shift_1
        self.bezier_points[p0 + 1].co.x -= x_shift_1
        self.bezier_points[p0 + 2].co.y += y_shift_2
        self.bezier_points[p0 + 2].co.x += x_shift_2
        # Make the X by swapping 1, 2
        self.bezier_points[p0 + 1].co.x, self.bezier_points[p0 + 2].co.x = self.bezier_points[p0 + 2].co.x, self.bezier_points[p0 + 1].co.x
        # Center the 4th point in the middle of 1, 2, randomize it slightly
        self.bezier_points[p0 + 4].co.x = (self.bezier_points[p0 + 1].co.x + self.bezier_points[p0 + 2].co.x)/2
        #self.bezier_points[p0 + 4].co.y = self.bezier_points[p0 + 1].co.y
        self.bezier_points[p0 + 4].co.y = 0.7*self.bezier_points[p0 + 1].co.y + 0.3*self.bezier_points[p0 + 3].co.y
        off = np.random.uniform(-0.15,0.15,3)
        off[2] = 0.0
        self.bezier_points[p0 + 4].co += Vector(tuple(off))

        # Raise the appropriate parts of the rope
        self.bezier_points[p0 + 1].co.z += 0.02 # TODO: sorry, this is a hack for now
        self.bezier_points[p0 + 3].co.z += 0.02

        # 5th point should be near the 4th point and raised up, randomized slightly
        self.bezier_points[p0 + 5].co = 0.5*self.bezier_points[p0+5].co + 0.5*self.bezier_points[p0+4].co
        #self.bezier_points[p0 + 5].co = self.bezier_points[p0+4].co 
        self.bezier_points[p0 + 5].co.z = 0.08
        off = np.random.uniform(-0.3,0.3,3)
        off[2] = 0.0
        self.bezier_points[p0 + 5].co += Vector(tuple(off))
        return set(range(p0, p0 + 5)) # this can be passed into offlimit_indices

    def make_simple_overlap(self, offset_min, offset_max):
        # Just makes a simple cubic loop
        # Geometrically arrange the bezier points into a loop, and slightly randomize over node positions for variety
        #    2_______1
        #     \     / 
        #      \   /
        #       \ /
        #       / \    
        #______0   3_____ 
        p0 = np.random.choice(range(len(self.bezier_points) - 5))
        y_shift = random.uniform(offset_min, offset_max)
        x_shift_1 = random.uniform(offset_min/3, offset_max/3)
        x_shift_2 = random.uniform(offset_min/3, offset_max/3)
        if random.uniform(0, 1) < 0.5:
            y_shift *= -1
        self.bezier_points[p0 + 1].co.y += y_shift
        self.bezier_points[p0 + 1].co.x -= x_shift_1
        self.bezier_points[p0 + 2].co.y += y_shift
        self.bezier_points[p0 + 2].co.x += x_shift_2
        self.bezier_points[p0 + 2].co.z += 0.06
        self.bezier_points[p0 + 1].co.x, self.bezier_points[p0 + 2].co.x = self.bezier_points[p0 + 2].co.x, self.bezier_points[p0 + 1].co.x
        return set(range(p0, p0 + 2)) # this can be passed into offlimit_indices

    def make_tangle(self, offset_min, offset_max):
        self.randomize_nodes(8, 0.20, 0.20)
        loop_indices = list(self.make_simple_loop(offset_min, offset_max))

    def randomize_nodes(self, num, x_offset_min, x_offset_max, y_offset_min, y_offset_max, nonplanar=False, offlimit_indices=set()):
        choices = list(set(range(len(self.bezier_points))) ^ offlimit_indices)
        knots_idxs = np.random.choice(choices, min(num, len(choices)), replace=False)
        for idx in knots_idxs:
            knot = self.bezier_points[idx]
            offset_x = random.uniform(x_offset_min, x_offset_max)
            offset_y = random.uniform(y_offset_min, y_offset_max)
            if random.uniform(0, 1) < 0.5:
                offset_y *= -1
            if random.uniform(0, 1) < 0.5:
                offset_x *= -1
            if nonplanar:
                offset_z = random.uniform(offset_min, offset_max)
                if random.uniform(0, 1) < 0.5:
                    offset_z *= -1
                knot.co.z += offset_z
            res_y = knot.co.y + offset_y
            res_x = knot.co.x + offset_x
            knot.co.y = res_y
            knot.co.x = res_x

    def reposition_camera(self):
        # Orient camera towards the rope
        bpy.context.scene.camera = self.camera
        height = self.camera.location.z
        angle1 = np.random.uniform(0, pi/12)*random.choice((1,-1))
        angle2 = np.random.uniform(0, pi/12)*random.choice((1,-1))
        angle3 = np.random.uniform(0, pi/4)*random.choice((1,-1))
        self.camera.rotation_euler = (angle1, angle2, angle3) 
        bpy.ops.view3d.camera_to_view_selected()
        self.camera.location.z += np.random.uniform(2, 2.5)
        #offset = height*sin(angle1)
        #self.camera.location.z -= offset

    def update(self, obj):
        # Call this method whenever you want the updated coordinates of an object after it has been deformed
        scene = bpy.context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_deformed = obj.evaluated_get(depsgraph)
        return obj_deformed

    def set_render_settings(self, engine, folder, filename, render_width=640, render_height=480):
        scene = bpy.context.scene
        scene.render.resolution_percentage = 100
        render_scale = scene.render.resolution_percentage / 100
        scene.render.resolution_x = render_width
        scene.render.resolution_y = render_height
        scene.render.engine = engine
        filename = "./{}/{}".format(folder, filename)
        if engine == 'BLENDER_WORKBENCH':
            scene.world.color = (1, 1, 1)
            scene.render.display_mode
            scene.render.image_settings.color_mode = 'RGB'
            scene.display_settings.display_device = 'None'
            scene.sequencer_colorspace_settings.name = 'XYZ'
            scene.render.image_settings.file_format='PNG'
        elif engine == "BLENDER_EEVEE":
            scene.eevee.taa_samples = 1
            scene.eevee.taa_render_samples = 1
        scene.render.resolution_percentage = 100
        render_scale = scene.render.resolution_percentage / 100
        scene.render.resolution_x = render_width
        scene.render.resolution_y = render_height
        return filename

    def annotate(self, frame, mapping, num_annotations):
        scene = bpy.context.scene
        rope_deformed = self.update(self.rope_asymm)
        vertices = [rope_deformed.matrix_world @ v.co for v in list(rope_deformed.data.vertices)[::len(list(rope_deformed.data.vertices))//num_annotations]] 
        render_size = (scene.render.resolution_x, scene.render.resolution_y)
        pixels = []
        for i in range(len(vertices)):
            v = vertices[i]
            camera_coord = bpy_extras.object_utils.world_to_camera_view(scene, bpy.context.scene.camera, v)
            pixel = [round(camera_coord.x * render_size[0]), round(render_size[1] - camera_coord.y * render_size[1])]
            pixels.append([pixel])
        mapping[frame] = pixels
        return mapping

    def render(self, filename, index, obj, annotations=None, num_annotations=0):
        scene = bpy.context.scene
        scene.render.filepath = filename % index
        bpy.ops.render.render(write_still=True)
        if annotations is not None:
            annotations = self.annotate(index, annotations, num_annotations)
        return annotations

    def render_dataset(self):
        if not os.path.exists("./images"):
            os.makedirs('./images')
        else:
            os.system('rm -r ./images')
            os.makedirs('./images')
        self.clear()
        self.add_camera(fixed=True)
        self.make_rigid_rope()
        if self.asymmetric:
            self.add_rope_asymmetry()
        self.make_bezier()
        if not self.save_rgb:
            engine = 'BLENDER_EEVEE'
        elif self.save_rgb:
            engine = 'BLENDER_WORKBENCH'
        render_path = self.set_render_settings(engine, 'images', '%06d_rgb.png')
        annot = {}
        for episode in range(self.num_images):
            self.straighten_rope()
            loop_indices = self.make_simple_loop(0.05, 0.075, 0.15, 0.5, p0=random.choice((3,4,5)))
            self.randomize_nodes(4, 0.05, 0.1, 0.05, 0.01, offlimit_indices = loop_indices)
            self.reposition_camera()
            annot = self.render(render_path, episode, self.rope_asymm, annotations=annot, num_annotations=self.num_annotations) # Render, save ground truth
        with open("./images/knots_info.json", 'w') as outfile:
            json.dump(annot, outfile, sort_keys=True, indent=2)

if __name__ == '__main__':
    with open("params.json", "r") as f:
        rope_params = json.load(f)
    renderer = RopeRenderer(save_depth=rope_params["save_depth"], 
                            asymmetric=rope_params["asymmetric"],
                            save_rgb=(not rope_params["save_depth"]),
                            num_images = rope_params["num_images"],
                            num_annotations=rope_params["num_annotations"],
                            bezier_knots=rope_params["bezier_knots"],
                            bezier_scale=rope_params["bezier_scale"],
                            nonplanar=rope_params["nonplanar"],
                            render_width=rope_params["render_width"],
                            render_height=rope_params["render_height"],
                            sequence=rope_params["sequence"],
                            episode_length=rope_params["episode_length"])
    renderer.render_dataset()
