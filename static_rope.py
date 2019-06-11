import sys

import direct.directbase.DirectStart
from direct.showbase.DirectObject import DirectObject
from direct.showbase.InputStateGlobal import inputState

from math import pi, sin, cos

from direct.task import Task

from panda3d.core import AmbientLight
from panda3d.core import DirectionalLight
from panda3d.core import Vec3
from panda3d.core import Vec4
from panda3d.core import CollisionNode, CollisionRay, CollisionHandlerQueue, CollisionTraverser
from panda3d.core import Point3
from panda3d.core import TransformState
from panda3d.core import BitMask32
from panda3d.core import GeomNode
from panda3d.core import RopeNode
from panda3d.core import NurbsCurveEvaluator

from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletDebugNode
from panda3d.bullet import BulletSoftBodyNode
from panda3d.bullet import BulletSoftBodyConfig

class Game(DirectObject):

  def __init__(self):
    base.setBackgroundColor(255, 255, 255)
    base.setFrameRateMeter(True)
#    base.disableMouse()

    base.cam.setPos(0, -50, 0)
    base.cam.lookAt(0, 0, 0)
    # base.disableMouse()

    base.useDrive()
    base.useTrackball()

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

    # Physics
    self.setup()

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

    self.world.doPhysics(dt, 10, 0.004)

    return task.cont

  def cleanup(self):
    self.world = None
    self.worldNP.removeNode()

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
    def make(p1):
      n = 8
      p2 = p1 + Vec3(0, 0, -12)

      bodyNode = BulletSoftBodyNode.makeRope(info, p1, p2, n, 3)
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
      visNP.setTexture(loader.loadTexture('assets/bowl.jpg'))

      #bodyNP.showBounds()
      #visNP.showBounds()

      return bodyNP

    np1 = make(Point3(-2, -1, 8))
#      np1 = make(Point3(2, 0, 0))

    # Box
    shape = BulletBoxShape(Vec3(2, 2, 2))

    boxNP = self.worldNP.attachNewNode(BulletRigidBodyNode('Box'))
    boxNP.node().setMass(50.0)
    boxNP.node().addShape(shape)
    boxNP.setPos(-2, -1, -5)
    boxNP.setCollideMask(BitMask32.allOn())

    self.world.attachRigidBody(boxNP.node())

    np1.node().appendAnchor(np1.node().getNumNodes() - 1, boxNP.node())

    visNP = loader.loadModel('assets/box.egg')
    visNP.clearModelNodes()
    visNP.setScale(4, 4, 4)
    visNP.reparentTo(boxNP)
    render.ls()

game = Game()
run()
