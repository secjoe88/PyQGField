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
            if len(points.shape)!=1 and not shape(points)[1]==3:
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
            if len(vectors.shape)!=1 and not shape(vectors)[1]==3:
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
        
        #fetch vectors, transformation matrix, creat temp matrix
        vecs=vstack([append(vector,1.).astype(float32) for vector in vectors])
        transMat=vstack([pg.transformToArray(arrow.transform().inverted()[0]) for arrow in self.arrows]).astype(float32)
        tempMat=empty_like(transMat)
        log.debug(
            'Fetched vectors, transformation matrices as:\nVecs:\n%s\n...\nTM\n%s\n...',
            str(vecs[0:3]),
            str(transMat)
        ) if self.logged else None
        
        #allocate gpu mem for vectors, transformation matrix, temp matrix
        vecs_gpu=cuda.mem_alloc(vecs.nbytes)
        transMat_gpu=cuda.mem_alloc(transMat.nbytes)
        tempMat_gpu=cuda.mem_alloc(tempMat.nbytes)
        
        #copy vectors, transformation matrix,to allocated gpu memory
        cuda.memcpy_htod(vecs_gpu, vecs)
        cuda.memcpy_htod(transMat_gpu, transMat)
        
        #log.debug('Empty temp matrix:\n%s\n...',str(tempMat))
        
        #CUDA kernel module
        mod=SourceModule("""
            __global__ void mFP1(float pvec[][4], float transMat[][4], float tempMat[][4]){
                 int i=threadIdx.x; int j=threadIdx.y;
                 
                tempMat[i][j]+=transMat[i][j]*pvec[i/4][j];
            }
            __global__ void mFP2(float tempMat[][4], float lvec[][4]){
                int i=threadIdx.x;
                for (int j=0;j<4;j++)
                    lvec[i/4][i%4]+=tempMat[i][j];
            }
        """)
        
        #get and run gpu function to map from parent to local coordinates
        log.info('Mapping parent vector to local vector coordinates...')
        log.debug('Attempting cuda function compile...') if self.logged else None
        mFP1=mod.get_function("mFP1")
        mFP2=mod.get_function("mFP2")
        log.debug(
            'Cuda function compiled! Attempting cuda function run with blocksize %dx4...',
            len(transMat)) if self.logged else None
        lvecs_gpu=cuda.mem_alloc(vecs.nbytes)
        mFP1(vecs_gpu,transMat_gpu,tempMat_gpu,block=(len(transMat),4,1))
        
        #log temp matrix
        cuda.memcpy_dtoh(tempMat,tempMat_gpu)
        log.debug('Augmented temp matrix:\n%s\n...',str(tempMat))
        mFP2(tempMat_gpu,lvecs_gpu,block=(len(transMat),1,1))
        log.debug('Cuda function ran!') if self.logged else None
        
        #retrieve results from gpu memory
        log.info('Extracting result from gpu memory...') if self.logged else None
        lvecs=empty_like(vecs)
        log.debug('Created container for vectors in local memory as:\n%s\n...',str(lvecs[0:3])) if self.logged else None
        cuda.memcpy_dtoh(lvecs,lvecs_gpu)
        log.debug('Extracted results from gpu memory as:\n%s\n...',str(lvecs[0:3])) if self.logged else None
        log.info('Successful map')
    
    
    
    
    
    
    
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
                
            
    
            
