import bpy
from mathutils import *; from math import *
import time
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors

# sample data
pixels, coords = [list(i) for i in np.loadtxt('images_goal/pixs.txt')], np.loadtxt('images_goal/coords.txt') # load long list of world co, pixel co candidate pairs
M_pix = 10
neigh = NearestNeighbors(1, M_pix)
neigh.fit(pixels)
print(pixels)
print(coords)
final_coords = [] # stores world coordinates of predicted pixels
# pixels_pred = list(sorted([list(i) for i in np.loadtxt('pixels_pred.txt')], key=lambda p: p[1]))
pixels_pred = [list(i) for i in np.loadtxt('pixels_pred.txt')]
for p in pixels_pred:
    print(p)
    match_idx = neigh.kneighbors([p], 1, return_distance=False).squeeze()
    print(pixels[match_idx])
    final_coords.append(Vector((coords[match_idx][0], coords[match_idx][1], coords[match_idx][2])))

bezier = bpy.data.objects['Bezier']
bezier_points = list(bezier.data.splines[0].bezier_points)
vertex_coords = [bezier.matrix_world.inverted() @ v for v in final_coords] # Get world coordinates in bezier frame of reference

bpy.context.view_layer.objects.active = bezier

render_count = 0
norm_diffs = float('inf')
scene = bpy.context.scene

os.system('rm -rf ./anim')
os.makedirs('./anim')

while norm_diffs > 0.001:
    print(norm_diffs)
    ind = sorted(range(len(vertex_coords)), key=lambda i: -np.linalg.norm(bezier_points[i].co - vertex_coords[i]))[0]
    bezier_points[ind].co = vertex_coords[ind]
    filename = "{0:06d}_anim.png".format(render_count)
    scene.world.color = (1, 1, 1)
    scene.render.display_mode
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display_settings.display_device = 'None'
    scene.sequencer_colorspace_settings.name = 'XYZ'
    scene.render.image_settings.file_format='PNG'
    scene.render.filepath = "./anim/{}".format(filename)
    bpy.ops.render.render(use_viewport = True, write_still=True)
    render_count += 1
    norm_diffs = sum(np.linalg.norm(bezier_points[i].co - vertex_coords[i]) for i in range(len(vertex_coords)))
