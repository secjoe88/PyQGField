import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pyqgfield.items import GLArrowItem
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from numpy import *
import pyqtgraph as pg
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
        ha                  bool determining if this quiver does hardware accelerated calculations
        ============== =====================================================
        """
        self.arrows=[]
        GLGraphicsItem.__init__(self)
        self.logged=kwds.pop('logged') if ('logged' in kwds) else False
        self.ha=kwds.pop('ha') if ('ha' in kwds) else False
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
            
            if self.ha:
                #use hardware acceleration if it is enabled
                self._haSet(vectors)
                end=time.time()-start
            else:
                #otherwise update vectors in software
                i=0
                for vector in vectors:
                    curTimes=self.arrows[i].updateData(vector=vector)
                    for opt in ['mapToLocal','cross','calcAngle','rotate','scl']:
                        times[opt]+=curTimes.pop(opt)*1000
                    i+=1
                end=time.time()-start
                #log specifics to file to file
                for opt in times:
                    log.debug('%d %s steps calculated in %d ms',len(self.arrows),opt,times[opt])
                
            #log elapsed time to file        
            log.info('%d arrows updated in %f ms',len(self.arrows),end*1000) if self.logged else None
        
        #update view
        self.update()
        return times
            
    #experimental function for hardware accelerated updates
    def _haSet(self, vectors=[]):
        log.info('Attempting hardware accelerated setting...')
        
        #extract arrow vectors from self.arrows and put into a gpu vector
        vecs=vstack([append(arrow.getVector(),1).astype(float32) for arrow in self.arrows])
        log.debug('Extracted vectors from self.arrows. Vecs:\n%s\n...',str(vecs[0:3])) if self.logged else None
        vecs_gpu=cuda.mem_alloc(vecs.nbytes)
        cuda.memcpy_htod(vecs_gpu, vecs)
        
        #extract transformation matrix and put into gpu vector
        transMat=pg.transformToArray(self.viewTransform().inverted()[0]).astype(float32)
        log.debug('Extracted inverted transformation matrix as:\n%s',str(transMat)) if self.logged else None
        transMat_gpu=cuda.mem_alloc(transMat.nbytes)
        cuda.memcpy_htod(transMat_gpu, transMat)
        
        #allocate gpu memory for output lvec
        lvecs_gpu=cuda.mem_alloc(zeros([len(vecs),3]).astype(float32).nbytes)
        #CUDA kernel module
        mod=SourceModule("""
            __global__ void mapFromParent(float pvec[][4], float mat[4][4], float lvec[][3]){
                int idx=threadIdx.x;
                float temp[4];
                for (int i=0; i<4; i++){
                    temp[i]=mat[i][0]*pvec[idx][0]+mat[i][1]*pvec[idx][1]+mat[i][2]*pvec[idx][2]+mat[i][3]*pvec[idx][3];
                }
                for (int i=0;i<3;i++)
                    lvec[idx][i]=temp[i];
            }
        """)
        #get and run gpu function to map from parent to local coordinates
        log.info('Mapping parent vector to local vector coordinates...')
        log.debug('Attempting cuda function compile...') if self.logged else None
        mapFP=mod.get_function("mapFromParent")
        log.debug('Cuda function compiled! Attempting cuda function run...') if self.logged else None
        mapFP(vecs_gpu,transMat_gpu,lvecs_gpu,block=(len(vecs),1,1))
        log.debug('Cuda function ran!') if self.logged else None
        
        #retrieve results from gpu memory
        log.info('Extracting result from gpu memory...') if self.logged else None
        lvecs=empty([len(vecs),3]).astype(float32)
        log.debug('Created container for vectors in local memory as:\n%s\n...',str(lvecs[0:3])) if self.logged else None
        cuda.memcpy_dtoh(lvecs,lvecs_gpu)
        log.debug('Extracted results from gpu memory as:\n%s\n...',str(lvecs[0:3])) if self.logged else None
        log.info('Succesful map')
    
    
    
    
    
    
    
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
                
            
    
            
