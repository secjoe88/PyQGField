import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pyqgfield.items import GLArrowItem
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from numpy import *
import pyqgfield
import logging
import time

__all__=['GLQuiverItem']
log=logging.getLogger(__name__)

class GLQuiverItem(GLGraphicsItem):
    def __init__(self, **kwds):
        """
        ============== =====================================================
        Arguments
        points              (N,3) array points[i]=[xi,yi,zi] specifying where to place
                            vectors
        vectors             (N,3) array vectors[i]=[vx_i,vy_i,vz_i] specifying the
                            vector placed at pts[i]
        logged              bool determining if this session be logged to file
        ============== =====================================================
        """
        self.arrows=[]
        GLGraphicsItem.__init__(self)
        self.logged=kwds.pop('logged') if ('logged' in kwds) else False
        self.setData(**kwds)
        
        
        
    def setData(self, **kwds):
        if len(kwds)==0:
            return
        if 'points' in kwds:
            points=array(kwds.pop('points'))
            #make sure  points is (N,3)
            if not shape(points)[1]==3:
                print('Points must be (N,3) array.')
                return
            #if self.arrows isn't instantiated, instantiate it from points, and set 
            #parent item to this instance of quiverItem
            elif not len(self.arrows)==len(points):
                self.arrows=[GLArrowItem() for i in range(len(points))]
                for arrow in self.arrows:
                    arrow.setParentItem(self)
            #update point values in self.arrows
            i=0
            for point in points:
                self.arrows[i].updateData(point=point)
                i+=1
        if 'vectors' in kwds:
            times={
                'mapToLocal': 0.,
                'cross':0.,
                'calcAngle':0.,
                'rotate':0.,
                'scl':0.
            }
            vectors=array(kwds.pop('vectors'))
            #make sure vectors is (N,3)
            if not shape(vectors)[1]==3:
                print('Vectors must be (N,3) array')
                return
            #make sure vectors is same length as self.arrows
            elif not len(self.arrows)==len(vectors):
                print(
                    'Expected vector argument with ' + str(len(self.arrows))
                    +' rows, but got argument with '+ str(len(vectors)) + ' rows'
                )
                return
            #update vector values in self.arrows
            log.info('Updating values for %d arrows...',len(self.arrows)) if self.logged else None
            start=time.time()
            i=0
            for vector in vectors:
                curTimes=self.arrows[i].updateData(vector=vector)
                for opt in ['mapToLocal','cross','calcAngle','rotate','scl']:
                    times[opt]+=curTimes.pop(opt)*1000
                i+=1
            end=time.time()-start
            for opt in times:
                log.debug('%d %s steps calculated in %d ms',len(self.arrows),opt,times[opt]*1000)
            log.info('%d arrows updated in %f ms',len(self.arrows),end*1000) if self.logged else None
        #update view
        self.update()
        
            
    
    def _haSet(self, vectors=[]):
        mod=SourceModule("""
            __global__ void transform()
        """)
    
    
    
    
    
    
    
    def random(self, shells):
        th=pi*random.random()
        phi=2*pi*random.random()
        pts=array([2*sin(th)*cos(phi),2*sin(th)*sin(phi),2*cos(th)])

        for i in range(shells):
            r=2*(i+1)
            for j in range(int(floor((4*pi*r**2)/2))):
                th=pi*random.random()
                phi=2*pi*random.random()
                pts=vstack([pts,[r*sin(th)*cos(phi),r*sin(th)*sin(phi),r*cos(th)]])

        self.setData(points=pts, vectors=-pts)
        return pts,-pts
                
            
    
            
