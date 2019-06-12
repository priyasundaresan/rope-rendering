import sys
from math import pi, sin, cos
import random
import numpy as np
import time
import cv2

import direct.directbase.DirectStart
from direct.showbase.DirectObject import DirectObject

from panda3d.core import AmbientLight, DirectionalLight, FrameBufferProperties, WindowProperties, \
Filename, Vec3, Vec4, CollisionNode, CollisionRay, CollisionHandlerQueue, CollisionTraverser, \
Point3, TransformState, BitMask32, GeomNode, RopeNode, NurbsCurveEvaluator, GraphicsPipe, GraphicsOutput, \
Texture, loadPrcFileData

from panda3d.bullet import BulletWorld, BulletPlaneShape, BulletBoxShape, BulletRigidBodyNode, \
BulletDebugNode, BulletSoftBodyNode, BulletSoftBodyConfig

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--static', action='store_true')
parser.add_argument('--random', action='store_true')
args = parser.parse_args()

def show_rgbd_image(image, depth_image, window_name='Image window', delay=1, depth_offset=None, depth_scale=0.009):
    if depth_image.dtype != np.uint8:
        if depth_scale is None:
            depth_scale = depth_image.max() - depth_image.min()
            print(depth_scale)
        if depth_offset is None:
            depth_offset = depth_image.min()
        depth_image = np.clip((depth_image - depth_offset) / depth_scale, 0.0, 1.0)
        depth_image = (255.0 * depth_image).astype(np.uint8)
    depth_image = np.tile(depth_image, (1, 1, 3))
    if image.shape[2] == 4:  # add alpha channel
        alpha = np.full(depth_image.shape[:2] + (1,), 255, dtype=np.uint8)
        depth_image = np.concatenate([depth_image, alpha], axis=-1)
        depth_image = cv2.bitwise_not(depth_image)
    images = np.concatenate([image, depth_image], axis=1)
#    images = cv2.bitwise_not(images)
    # images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)  # not needed since image is already in BGR format
    cv2.imshow(window_name, images)
    key = cv2.waitKey(delay)
    key &= 255
    if key == 27 or key == ord('q'):
        print("Pressed ESC or q, exiting")
        exit_request = True
    else:
        exit_request = False
    return exit_request

def randomDegree():
    return np.random.randint(361)

def degreesToRadians(degree):
    return degree * (pi / 180)

class Game(DirectObject):

  def __init__(self):
    self.scene_limit = 200
    self.scene_curr = 0
    base.setBackgroundColor(255, 255, 255)
    base.setFrameRateMeter(True)

    base.cam.setPos(0, -40, 10)
    base.cam.lookAt(0, 0, 0)

    # Light
    alight = AmbientLight('ambientLight')
    alight.setColor(Vec4(0.5, 0.5, 0.5, 1))
    alightNP = render.attachNewNode(alight)

    dlight = DirectionalLight('directionalLight')
    dlight.setDirection(Vec3(5, 0, -2))
    dlight.setColor(Vec4(0.7, 0.7, 0.7, 1))
    dlightNP = render.attachNewNode(dlight)

    render.clearLight()
    render.setLight(alightNP)
    render.setLight(dlightNP)

    # Input
    self.accept('escape', self.doExit)
    self.accept('r', self.doReset)
    self.accept('w', self.toggleWireframe)
    self.accept('t', self.toggleTexture)
    self.accept('d', self.toggleDebug)
    self.accept('s', self.doScreenshot)

    # Task
    taskMgr.add(self.update, 'updateWorld')
    taskMgr.add(self.operateCamera, 'operateCamera')

    # Physics
    self.setup()

    # camera stuff
    # Needed for camera image
    self.dr = base.camNode.getDisplayRegion(0)
    # Needed for camera depth image
    winprops = WindowProperties.size(base.win.getXSize(), base.win.getYSize())
#    winprops = WindowProperties.size(640, 480)
    fbprops = FrameBufferProperties()
    fbprops.setDepthBits(1)
    self.depthBuffer = base.graphicsEngine.makeOutput(
        base.pipe, "depth buffer", -2,
        fbprops, winprops,
        GraphicsPipe.BFRefuseWindow,
        base.win.getGsg(), base.win)
    self.depthTex = Texture()
    self.depthTex.setFormat(Texture.FDepthComponent)
    self.depthBuffer.addRenderTexture(self.depthTex,
        GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPDepth)
    lens = base.camNode.getLens()
    # the near and far clipping distances can be changed if desired
    # lens.setNear(5.0)
    # lens.setFar(500.0)
    self.depthCam = base.makeCamera(self.depthBuffer,
        lens=lens,
        scene=render)
    self.depthCam.reparentTo(base.cam)

  def get_camera_image(self, requested_format=None):
      """
      Returns the camera's image, which is of type uint8 and has values
      between 0 and 255.
      The 'requested_format' argument should specify in which order the
      components of the image must be. For example, valid format strings are
      "RGBA" and "BGRA". By default, Panda's internal format "BGRA" is used,
      in which case no data is copied over.
      """
      tex = self.dr.getScreenshot()
      if requested_format is None:
          data = tex.getRamImage()
      else:
          data = tex.getRamImageAs(requested_format)
      image = np.frombuffer(data, np.uint8)  # use data.get_data() instead of data in python 2
      image.shape = (tex.getYSize(), tex.getXSize(), tex.getNumComponents())
      image = np.flipud(image)
      return image

  def get_camera_depth_image(self):
      """
      Returns the camera's depth image, which is of type float32 and has
      values between 0.0 and 1.0.
      """
      data = self.depthTex.getRamImage()
      depth_image = np.frombuffer(data, np.float32)
      depth_image.shape = (self.depthTex.getYSize(), self.depthTex.getXSize(), self.depthTex.getNumComponents())
      depth_image = np.flipud(depth_image)
      return depth_image


  # _____HANDLER_____

  def doExit(self):
    self.cleanup()
    sys.exit(1)

  def doReset(self):
    self.cleanup()
    self.setup()

  def toggleWireframe(self):
    base.toggleWireframe()

  def toggleTexture(self):
    base.toggleTexture()

  def toggleDebug(self):
    if self.debugNP.isHidden():
      self.debugNP.show()
    else:
      self.debugNP.hide()

  def doScreenshot(self):
    base.screenshot('Bullet')

  # ____TASK___

  def update(self, task):
    dt = globalClock.getDt()
   # print(base.cam.getHpr())
    self.world.doPhysics(dt, 10, 0.004)
    # if self.scene_curr < self.scene_limit:
    #     if self.scene_curr % 10 == 0:
    #         # base.win.saveScreenshot(Filename("{0:06d}_rgb.png".format(self.scene_curr//10)))
    #     self.scene_curr += 1
    # else:
    #     self.doExit()

    return task.cont

  def cleanup(self):
    self.world = None
    self.worldNP.removeNode()

  def operateCamera(self, task):
        # Rotates the camera and takes RGBD images
        if args.random:
         #  base.cam.setPos(0.0, -40, 0)
           base.cam.setPos(40.0 * sin(degreesToRadians(randomDegree())), -40.0 * cos(degreesToRadians(randomDegree())), 10 * cos(degreesToRadians(randomDegree())))
           base.cam.setHpr(randomDegree(), randomDegree(), randomDegree())
           base.cam.lookAt(random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0))
        else:
           angleDegrees = task.time * 20.0
           angleRadians = angleDegrees * (pi / 180.0)
           base.cam.setPos(40 * sin(angleRadians), -40.0 * cos(angleRadians), 10)
           base.cam.setHpr(angleDegrees, 0, 0)
           base.cam.lookAt(0, 0, 0)
        base.graphicsEngine.renderFrame()
        I, J, K, R = base.cam.getQuat().getI(), \
        base.cam.getQuat().getJ(), \
        base.cam.getQuat().getK(), \
        base.cam.getQuat().getR()
        # print(base.cam.getQuat())
        quaternion = {'w': K, 'x': R, 'y': I, 'z': J}
        print(quaternion)
        print(base.cam.getPos())
        image = self.get_camera_image()
        depth_image = self.get_camera_depth_image()
        show_rgbd_image(image, depth_image)
        return task.cont

  def setup(self):
    self.worldNP = render.attachNewNode('World')

    # World
    self.debugNP = self.worldNP.attachNewNode(BulletDebugNode('Debug'))
    self.debugNP.show()

    self.world = BulletWorld()
    self.world.setGravity(Vec3(0, 0, -9.81))
    self.world.setDebugNode(self.debugNP.node())

    # Soft body world information
    info = self.world.getWorldInfo()
    info.setAirDensity(1.2)
    info.setWaterDensity(0)
    info.setWaterOffset(0)
    info.setWaterNormal(Vec3(0, 0, 0))

    # Softbody
    def make(p1, offset, fixed):
      n = 8 
      p2 = p1 + offset

      bodyNode = BulletSoftBodyNode.makeRope(info, p1, p2, n, fixed)
      bodyNode.setTotalMass(50.0)
      bodyNP = self.worldNP.attachNewNode(bodyNode)
      self.world.attachSoftBody(bodyNode)

      # Render option 1: Line geom
      #geom = BulletSoftBodyNode.makeGeomFromLinks(bodyNode)
      #bodyNode.linkGeom(geom)
      #visNode = GeomNode('')
      #visNode.addGeom(geom)
      #visNP = bodyNP.attachNewNode(visNode)

      # Render option 2: NURBS curve
      curve = NurbsCurveEvaluator()
      curve.reset(n + 2)
      bodyNode.linkCurve(curve)

      visNode = RopeNode('Rope')
      visNode.setCurve(curve)
      visNode.setRenderMode(RopeNode.RMTube)
      visNode.setUvMode(RopeNode.UVParametric)
      visNode.setNumSubdiv(4)
      visNode.setNumSlices(8)
      visNode.setThickness(0.4)
      visNP = self.worldNP.attachNewNode(visNode)
      #visNP = bodyNP.attachNewNode(visNode) # --> renders with offset!!!
      visNP.setTexture(loader.loadTexture('assets/simple.jpg'))

      #bodyNP.showBounds()
      #visNP.showBounds()

      return bodyNP

    shape = BulletBoxShape(Vec3(0.5, 0.5, 0.5))

    boxNP = self.worldNP.attachNewNode(BulletRigidBodyNode('Box'))
    boxNP.node().setMass(50.0)
    boxNP.node().addShape(shape)
    boxNP.setCollideMask(BitMask32.allOn())

    if args.static:
    	np1 = make(Point3(-2, -1, 8), Vec3(0, 0, -12), 3)
    	boxNP.setPos(-2, -1, -4)
    else:
    	np1 = make(Point3(-2, -1, 8), Vec3(12, 0, 0), 1)
    	boxNP.setPos(10, -1, 8)
    	

    # Box
    self.world.attachRigidBody(boxNP.node())

    np1.node().appendAnchor(np1.node().getNumNodes() - 1, boxNP.node())

    visNP = loader.loadModel('assets/box.egg')
    visNP.clearModelNodes()
    visNP.setScale(1, 1, 1)
    visNP.reparentTo(boxNP)
    render.ls()

game = Game()

run()
