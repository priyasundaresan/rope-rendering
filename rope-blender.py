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
    def __init__(self, asymmetric=False, rope_radius=None, sphere_radius=None, rope_iterations=None, rope_screw_offset=None, bezier_scale=3.7, bezier_knots=12, save_depth=True, save_rgb=False, num_annotations=20, num_images=10, nonplanar=True, render_width=640, render_height=480, sequence=False, episode_length=1, domain_randomize=False):
        self.save_rgb = save_rgb # whether to save_rgb images or not
        self.num_images = num_images
        self.sequence = sequence
        self.episode_length = episode_length
        self.render_width = render_width
        self.render_height = render_height
        self.num_annotations = num_annotations
        self.save_depth = save_depth
        self.domain_randomize = domain_randomize

        self.asymmetric = asymmetric
        self.rope_radius = rope_radius
        self.rope_screw_offset = rope_screw_offset
        self.sphere_radius = sphere_radius
        self.rope_iterations = rope_iterations
        self.nonplanar = nonplanar
        #self.bezier_scale = None
        self.bezier_scale = bezier_scale
        self.bezier_subdivisions = bezier_knots - 2 # the number of splits in the bezier curve (ctrl points - 2)
        self.origin = (0, 0, 0)
        # Make objects
        self.rope = None
        self.rope_asymm = None
        self.bezier = None
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

    def pattern(self, obj, texture_filename):
        '''Add image texture to object'''
        if '%sTexture' % obj.name in bpy.data.materials:
            mat = bpy.data.materials['%sTexture'%obj.name]
        else:
            mat = bpy.data.materials.new(name="%sTexture"%obj.name)
            mat.use_nodes = True
        texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        texImage.image = bpy.data.images.load(texture_filename)
        mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
        mat.specular_intensity = np.random.uniform(0, 0.3)
        mat.roughness = np.random.uniform(0.5, 1)
        if not obj.data.materials:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat

    def set_viewport_shading(self, mode):
        '''Makes color/texture viewable in viewport'''
        areas = bpy.context.workspace.screens[0].areas
        for area in areas:
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = mode

    def texture_randomize(self, obj, textures_folder):
        rand_img_path = random.choice(os.listdir(textures_folder))
        img_filepath = os.path.join(textures_folder, rand_img_path)
        self.pattern(obj, img_filepath)
    
    def colorize(self, obj, color):
        '''Add color to object'''
        if '%sColor' % obj.name in bpy.data.materials:
            mat = bpy.data.materials['%sColor'%obj.name]
        else:
            mat = bpy.data.materials.new(name="%sColor"%obj.name)
            mat.use_nodes = False
        mat.diffuse_color = color
        mat.specular_intensity = np.random.uniform(0, 0.1)
        mat.roughness = np.random.uniform(0.5, 1)
        if not obj.data.materials:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat

    def make_table(self):
        # Generate table surface
        bpy.ops.mesh.primitive_plane_add(size=15, location=(0,0,-0.1))
        #bpy.ops.transform.resize(value=(0.33, 0.33, 1))
        #bpy.ops.object.modifier_add(type='COLLISION')
        self.table = bpy.context.object

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

    def add_camera(self):
        '''
        Place a camera randomly, fixed means no rotations about z axis (planar camera changes only)
        '''
        bpy.ops.object.camera_add(location=[0, 0, 10])
        self.camera = bpy.context.active_object
        self.camera.rotation_euler = (0, 0, random.uniform(pi/4, 3*pi/4)) # fixed z, rotate only about x/y axis slightly
        self.camera.name = self.camera_name
        bpy.ops.object.light_add(type='SUN', radius=1, location=(0, 0, 0))

    def add_light(self):
        light_data = bpy.data.lights.new(name="Light", type='POINT')
        #light_data.energy = 300
        light_object = bpy.data.objects.new(name="LightObj", object_data=light_data)
        bpy.context.collection.objects.link(light_object)
        bpy.context.view_layer.objects.active = light_object
        light_object.location = (0, 0, 5)
        scene = bpy.context.scene
        #scene.view_settings.exposure = 1.5 

    def randomize_light(self):
        scene = bpy.context.scene
        scene.view_settings.exposure = random.uniform(0.5, 1.5)
        light_data = bpy.data.lights['Light']
        light_data.color = tuple(np.random.uniform(0,1,3))
        light_data.energy = np.random.uniform(20,200)
        light_data.shadow_color = tuple(np.random.uniform(0,1,3))
        light_obj = bpy.data.objects['LightObj']
        light_obj.data.color = tuple(np.random.uniform(0.7,1,3))

    def make_rigid_chord(self):
        '''
        Make one long cylinder
        '''
        bpy.ops.mesh.primitive_circle_add(location=self.origin)
        radius = np.random.uniform(0.05, 0.1)
        bpy.ops.transform.resize(value=(radius, radius, radius))
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.join()
        self.rope = bpy.context.active_object
        self.rope_asymm = self.rope
        self.rope.name = self.rope_name
        bpy.ops.object.modifier_add(type='SCREW')
        screw_offset = np.random.uniform(12, 13)
        self.rope.modifiers["Screw"].screw_offset = screw_offset
        rope_iterations = np.random.uniform(9.5, 10.5)
        self.rope.modifiers["Screw"].iterations = rope_iterations
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Screw")

    def make_rigid_chain(self):
        '''
        Join multiple toruses together to make chain shape (http://jayanam.com/chains-with-blender-tutorial/)
        TODOS: fix torus vertex selection to make oval shape
               fix ARRAY selection after making empty mesh
        '''
        # hacky fix from https://www.reddit.com/r/blenderhelp/comments/dnb56f/rendering_python_script_in_background/
        for window in bpy.context.window_manager.windows:
            screen = window.screen
            for area in screen.areas:
                if area.type == 'VIEW_3D':
                    override = {'window': window, 'screen': screen, 'area': area}
                    bpy.ops.screen.screen_full_area(override)
                    break
        bpy.ops.mesh.primitive_torus_add(align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), major_radius=1, minor_radius=0.25, abso_major_rad=1.25, abso_minor_rad=0.75)
        bpy.ops.object.editmode_toggle()
        bpy.ops.transform.translate(value=(0, -1, 0), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
        bpy.ops.transform.translate(value=(0, -1, 0), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
        bpy.ops.object.editmode_toggle()
        self.rope = bpy.context.active_object
        self.rope_asymm = self.rope
        self.rope.name = self.rope_name

        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 1.8, 0))
        bpy.ops.transform.rotate(value=1.5708, orient_axis='Y', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
        bpy.context.active_object.select_set(False)

        self.rope_asymm.select_set(True)
        bpy.ops.object.modifier_add(type='ARRAY')
        bpy.context.object.modifiers["Array"].use_relative_offset = False
        bpy.context.object.modifiers["Array"].use_object_offset = True
        bpy.context.object.modifiers["Array"].offset_object = bpy.data.objects["Empty"]

        # bpy.ops.object.modifier_add(type='CURVE')
        # bpy.context.object.modifiers["Curve"].object = bpy.data.objects["NurbsPath"] # FIX This
        # bpy.context.object.modifiers["Curve"].deform_axis = 'POS_Y'
        # bpy.context.object.modifiers["Array"].count = 20

    def make_rigid_rope(self):
        '''
        Join 4 circles and "twist" them to create realistic rope (See this 5 min. tutorial: https://youtu.be/xYhIoiOnPj4 if interested)
        '''
        bpy.ops.mesh.primitive_circle_add(location=self.origin)
        radius = np.random.uniform(0.048, 0.048) if self.rope_radius is None else self.rope_radius
        bpy.ops.transform.resize(value=(radius, radius, radius))
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.transform.rotate(value= pi / 2, orient_axis='X')
        bpy.ops.transform.translate(value=(radius, 0, 0))
        bpy.ops.object.mode_set(mode='OBJECT')
        num_chords = np.random.randint(3, 5)
        for i in range(1, num_chords):
            bpy.ops.object.duplicate_move(OBJECT_OT_duplicate=None, TRANSFORM_OT_translate=None)
            ob = bpy.context.active_object
            # ob.rotation_euler = (0, 0, i * (2*pi / num_chords))
            ob.rotation_euler = (0, 0, i * (pi / 2))
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.join()
        self.rope = bpy.context.active_object

        self.rope_asymm = self.rope

        self.rope.name = self.rope_name
        bpy.ops.object.modifier_add(type='SCREW')
        screw_offset = np.random.uniform(5,13) if self.rope_screw_offset is None else self.rope_screw_offset
        self.rope.modifiers["Screw"].screw_offset = screw_offset
        rope_iterations = 17.7 * 13/screw_offset if self.rope_iterations is None else self.rope_iterations
        self.rope.modifiers["Screw"].iterations = rope_iterations
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Screw")

    def add_rope_asymmetry(self):
        '''
        Add sphere, to break symmetry of the rope
        '''
        bpy.ops.mesh.primitive_uv_sphere_add(location=(self.origin[0], self.origin[1], self.origin[2]))
        if self.sphere_radius is not None:
            sphere_radius = self.sphere_radius
        else:
            sphere_radius = np.random.uniform(0.35, 0.37)
        if self.domain_randomize:
            yellow = np.array([0.75, 0.9, 0.1])
            offset = np.random.uniform(-0.1, 0.1, 3)
            color = tuple(np.hstack((yellow+offset, np.array([1]))))
            self.colorize(bpy.context.object, color)
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
        # Center the 4th point in the middle of 1, 2
        self.bezier_points[p0 + 4].co.x = (self.bezier_points[p0 + 1].co.x + self.bezier_points[p0 + 2].co.x)/2
        # Put 4th point's y coordinate right below 1st point's y coordinate
        self.bezier_points[p0 + 4].co.y = self.bezier_points[p0 + 1].co.y + np.random.uniform(-0.05, -0.1)
        # Randomize 4th point slightly
        off = np.random.uniform(-0.03,0.03,3)
        off[2] = 0.0
        self.bezier_points[p0 + 4].co += Vector(tuple(off))

        # Raise the appropriate parts of the rope
        self.bezier_points[p0 + 1].co.z += 0.02 # TODO: sorry, this is a hack for now
        self.bezier_points[p0 + 3].co.z += 0.02

        # 5th point should be near the 4th point and raised up, randomized slightly
        self.bezier_points[p0 + 5].co = 0.5*self.bezier_points[p0+5].co + 0.5*self.bezier_points[p0+4].co
        self.bezier_points[p0 + 5].co.z = 0.08
        # Randomize 5th point slightly
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
        #self.camera.rotation_euler = (random.uniform(-pi/12, pi/12), random.uniform(-pi/12, pi/12), random.uniform(-pi/4, pi/4)) # fixed z, rotate only about x/y axis slightly
        self.camera.rotation_euler = (random.uniform(0, 0), random.uniform(0, 0), random.uniform(-pi/4, pi/4)) # fixed z, rotate only about x/y axis slightly
        #self.camera.rotation_euler = (random.uniform(-pi/24, pi/24), random.uniform(-pi/24, pi/24), random.uniform(-pi/4, pi/4)) # fixed z, rotate only about x/y axis slightly
        bpy.ops.view3d.camera_to_view_selected()
        #self.camera.location.z += np.random.uniform(3.3, 3.6)
        self.camera.location.z += np.random.uniform(2.5, 3)

    def update(self, obj):
        # Call this method whenever you want the updated coordinates of an object after it has been deformed
        scene = bpy.context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_deformed = obj.evaluated_get(depsgraph)
        return obj_deformed

    def set_render_settings(self, engine, folder, filename, render_width=640, render_height=480):
        self.set_viewport_shading('MATERIAL')
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
            scene.view_settings.view_transform = 'Raw'
        scene.render.resolution_percentage = 100
        render_scale = scene.render.resolution_percentage / 100
        scene.render.resolution_x = render_width
        scene.render.resolution_y = render_height
        return filename

    def annotate(self, frame, mapping, num_annotations):
        scene = bpy.context.scene
        rope_deformed = self.update(self.rope_asymm)
        verts = list(rope_deformed.data.vertices)
        idxs = np.round(np.linspace(0, len(verts) - 1, self.num_annotations)).astype(int)
        verts_sparsified = [verts[i] for i in idxs]
        vertices = [rope_deformed.matrix_world @ v.co for v in verts_sparsified]
        render_size = (scene.render.resolution_x, scene.render.resolution_y)
        pixels = []
        for i in range(len(vertices)):
            v = vertices[i]
            camera_coord = bpy_extras.object_utils.world_to_camera_view(scene, bpy.context.scene.camera, v)
            pixel = [round(camera_coord.x * render_size[0]), round(render_size[1] - camera_coord.y * render_size[1])]
            pixels.append([pixel])
        mapping[frame] = pixels
        return mapping

    def render(self, filename, index, obj, annotations=None, num_annotations=0, export_mask=True):
        scene = bpy.context.scene
        if export_mask:
            self.render_mask("image_masks/%06d_visible_mask.png", "images_depth/%06d_rgb.png", index)
        scene.render.filepath = filename % index
        bpy.ops.render.render(write_still=True)
        if annotations is not None:
            annotations = self.annotate(index, annotations, num_annotations)
        return annotations

    def render_mask(self, mask_filename, depth_filename, index):
        # NOTE: this method is still in progress
        scene = bpy.context.scene
        saved = scene.render.engine
        scene.render.engine = 'BLENDER_EEVEE'
        scene.eevee.taa_samples = 1
        scene.eevee.taa_render_samples = 1
        scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links
        render_node = tree.nodes["Render Layers"]
        norm_node = tree.nodes.new(type="CompositorNodeNormalize")
        inv_node = tree.nodes.new(type="CompositorNodeInvert")
        math_node = tree.nodes.new(type="CompositorNodeMath")
        math_node.operation = 'CEIL' # Threshold the depth image
        composite = tree.nodes.new(type = "CompositorNodeComposite")

        links.new(render_node.outputs["Depth"], inv_node.inputs["Color"])
        links.new(inv_node.outputs[0], norm_node.inputs[0])
        links.new(norm_node.outputs[0], composite.inputs["Image"])

        scene.render.filepath = depth_filename % index
        bpy.ops.render.render(write_still=True)

        links.new(norm_node.outputs[0], math_node.inputs[0])
        links.new(math_node.outputs[0], composite.inputs["Image"])

        saved_filepath = scene.render.filepath
        scene.render.filepath = mask_filename % index
        bpy.ops.render.render(write_still=True)
        # Clean up 
        scene.render.engine = saved
        scene.render.filepath = saved_filepath
        for node in tree.nodes:
            if node.name != "Render Layers":
                tree.nodes.remove(node)
        scene.use_nodes = False

    def render_dataset(self):
        if not os.path.exists("./images"):
            os.makedirs('./images')
        else:
            os.system('rm -r ./images')
            os.makedirs('./images')
        if self.domain_randomize:
            engine = 'BLENDER_EEVEE'
        elif self.save_rgb:
            engine = 'BLENDER_WORKBENCH'
        render_path = self.set_render_settings(engine, 'images', '%06d_rgb.png')
        annot = {}
        for i in range(self.num_images):
            x = time.time()
            if not self.sequence or (self.sequence and i%self.episode_length == 0):
                self.clear()
                self.add_camera()
                self.add_light()
                rope_texture = np.random.randint(1, 3)
                if rope_texture == 1:
                    self.make_rigid_rope()
                else:
                    self.make_rigid_chord()
                if self.domain_randomize:
                    color = tuple(np.concatenate((np.random.uniform(0.8, 1, 3), np.array([1]))))
                    self.colorize(self.rope_asymm, color)
                    #self.texture_randomize(self.rope_asymm, 'rope_textures') # UNCOMMENT for randomizing rope textures
                if self.asymmetric:
                    self.add_rope_asymmetry()
                self.make_bezier()
            if self.nonplanar:
                # Generate a split of loops, knots, and planar configs
                rand = np.random.uniform()
                if rand < 0.33:
                    loop_indices = self.make_simple_loop(0.1, 0.15, 0.2, 0.35, p0=random.choice((3,4,5)))
                    self.randomize_nodes(4, 0.05, 0.1, 0.05, 0.01, offlimit_indices = loop_indices)
                elif rand < 0.66:
                    loop_indices = self.make_simple_loop(0.025, 0.075, 0.05, 0.2, p0=random.choice((3,4,5)))
                    self.randomize_nodes(4, 0.05, 0.1, 0.05, 0.01, offlimit_indices = loop_indices)
                else:
                    self.randomize_nodes(4, 0.1, 0.3, 0.1, 0.3)
            else:
                # Generate only planar configs
                    if not self.sequence or (self.sequence and i%self.episode_length == 0):
                        self.randomize_nodes(4, 0.1, 0.3, 0.1, 0.3)
                    else:
                        self.randomize_nodes(1, 0.03, 0.05, 0.03, 0.05)
            if not self.sequence or (self.sequence and i%self.episode_length == 0):
                self.reposition_camera()
            self.make_table()
            if self.domain_randomize:
                self.randomize_light()
                if random.random() < 0.5:
                    self.texture_randomize(self.table, 'val2017')
                else:
                    color = tuple(np.concatenate((np.random.uniform(0.5, 1, 3), np.array([1]))))
                    self.colorize(self.table, color)
                
            annot = self.render(render_path, i, self.rope_asymm, annotations=annot, num_annotations=self.num_annotations) # Render, save ground truth
        with open("./images/knots_info.json", 'w') as outfile:
            json.dump(annot, outfile, sort_keys=True, indent=2)

if __name__ == '__main__':
    with open("params.json", "r") as f:
        rope_params = json.load(f)
    renderer = RopeRenderer(save_depth=rope_params["save_depth"],
                            domain_randomize=rope_params["domain_randomize"],
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
    #renderer.run()
    renderer.render_dataset()
