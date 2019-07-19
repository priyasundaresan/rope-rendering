import bpy
from mathutils import *; from math import *
import time
import numpy as np
import os

# sample data
coords = [Vector((-3.5510969161987305, 0.3658915162086487, 0.07000000774860382)),
 Vector((-2.806908369064331, 0.3841162919998169, -0.010823898948729038)),
 Vector((-2.800145387649536, -0.06685233116149902, -0.030320491641759872)),
 Vector((-3.178575277328491, -0.4295629858970642, 2.4610930182689117e-08)),
 Vector((-2.84277081489563, -0.5980896949768066, -0.06222281977534294)),
 Vector((-2.6518774032592773, -0.21869879961013794, 0.04000001400709152)),
 Vector((-2.3578031063079834, 0.26199233531951904, -2.4586915614577265e-08)),
 Vector((-1.956253170967102, 0.373394638299942, -0.0769551619887352)),
 Vector((-1.6326137781143188, 0.7516661882400513, 0.0007685869932174683)),
 Vector((-1.3446204662322998, 1.1952705383300781, -1.4828982841663674e-07)),
 Vector((-1.015438199043274, 1.3075778484344482, 4.917382767644085e-08)),
 Vector((-1.2637770175933838, 0.8082948923110962, 0.02469266764819622)),
 Vector((-1.48521888256073, 0.3938544690608978, -2.4586915614577265e-08)),
 Vector((-0.9218199253082275, 0.3658914864063263, -6.661338147750939e-16)),
 Vector((-0.3995960056781769, 0.33785659074783325, 0.06768233329057693)),
 Vector((0.04122765734791756, 0.3753409683704376, -0.022813046351075172)),
 Vector((0.617414653301239, 0.39563700556755066, 0.012321241199970245)),
 Vector((1.0498992204666138, 0.33976036310195923, 0.06308649480342865)),
 Vector((1.5859864950180054, 0.36618563532829285, -0.000710045627783984)),
 Vector((2.093489408493042, 0.4369889497756958, 0.02944929525256157))]


bezier = bpy.data.objects['Bezier']
bezier_points = list(bezier.data.splines[0].bezier_points)
vertex_coords = [bezier.matrix_world.inverted() @ v for v in coords]

bpy.context.view_layer.objects.active = bezier

render_count = 0
norm_diffs = float('inf')
scene = bpy.context.scene

os.system('rm -rf ./anim')
os.makedirs('./anim')
while norm_diffs > 0.2:
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
