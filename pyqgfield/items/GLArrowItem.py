from pyqtgraph.opengl import GLLinePlotItem, GLMeshItem
from numpy import *
import time
import logging
import pyqgfield

__all__=['GLArrowItem']
log=logging.getLogger(__name__)

class GLArrowItem(GLMeshItem):

    def __init__(self,**kwds):
        """
        ============== =====================================================
        Arguments
        point             (1,3) array [x,y,z] specifying point coordinates
        vector            (1,3) array [vx,vy,vz] specifying the vector at the given point
        logged            boolean specifying if arrow should log to file
        ============== =====================================================
        """
    #create arrowhead
	#first create vertex coordinates and faces for planar circle to use as arrowhead base
        vertexes=array([0,0,2./3.])
        faces=zeros([10,3],dtype=int32)
        for i in range(10):
            newV=[(1./6.)*cos(i*(pi/5.)),(1./6.)*sin(i*(pi/5.)),2./3.]
            vertexes=vstack([vertexes,array(newV)])
            faces[i]=[0,i+1,i+2]
        faces[9]=[0,10,1]
		
	#now to create arrowtip vertex and faces for cone to complete arrowhead
        vertexes=vstack([vertexes,array([0.,0.,1.])])
        faces=vstack([faces,faces])
        faces[10:,0]=11
    #create stem
        #first we draw a vertex at the origin, and a planar circle of vertices at the base of the
        #arrowhead
        vertexes=vstack([vertexes,[0,0,0]])
        for i in range(10):
            newV=[(1./18.)*cos(i*(pi/5.)),(1./18.)*sin(i*(pi/5.)),2./3.]
            vertexes=vstack([vertexes,array(newV)])
        #then we create faces in a thinner inverted cone shape to serve as stem
        for i in range(12,len(vertexes)-2):
            faces=vstack([faces,[12,i+1,i+2]])
        faces=vstack([faces, [12,22,13]])
    #Initialize item from superclass
        GLMeshItem.__init__(
            self,
            vertexes=vertexes,
            faces=faces,
            color=[1,0,0,1],
            shader='shaded',
            glOptions='opaque'
            )
        self.logged=kwds.pop('logged') if ('logged' in kwds) else False
        self.vector=array([0.,0.,1.])
        self.point=array([0.,0.,0.])
        if 'point' in kwds and 'vector' in kwds:
            self.updateData(**kwds)


    def updateData(self, **kwds):
        times={}
        if 'point' in kwds:
            self.logged=False
            log.info('Found point in args. Updating point...') if self.logged else None
            point=array(kwds.pop('point'))#updated point in parent coordinates
            log.debug(
                'Attempting to translate point from [%f,%f,%f](parent) to [%f,%f,%f](parent)',
                self.point[0],self.point[1], self.point[2],
                point[0],point[1],point[2]
                ) if self.logged else None
            #translate the item to the coordinates in **kwds
            dx=point[0]-self.point[0];dy=point[1]-self.point[1];dz=point[2]-self.point[2]
            log.debug('Translating x by %f, y by %f, z by %f...',dx,dy,dz) if self.logged else None
            self.translate(dx=dx,dy=dy,dz=dz)
            #update current point
            self.point=point
            log.info('Finished attempt at moving point') if self.logged else None
        if 'vector' in kwds:
            self.logged=True
            log.info('Found vector in args. Attempting to transform vector (software)...') if self.logged else None
            pvector=array(kwds.pop('vector'))#updated vector in parent coordinates
            if not linalg.norm(pvector)==0:
                
                start=time.time()
                #map pvector to item's local coordinates and calculate rotation axis using cross product
                lvector=self.mapFromParent(vstack(pvector))
                lvector=lvector.flatten()/linalg.norm(lvector) #normalize
                times['mapToLocal']=time.time()-start
                
                start=time.time()
                axis=cross(array([0.,0.,1.]),lvector)
                times['cross']=time.time()-start
                
                start=time.time()
                #calculate rotation angle from cross product result accounting for angles that are >90
                angle=0.
                if linalg.norm([lvector-array([0.,0.,1.])])>sqrt(2):
                    angle=180.-arcsin(linalg.norm(axis)/(linalg.norm(lvector)))*(180./pi)
                else:
                    angle+=arcsin(linalg.norm(axis)/(linalg.norm(lvector)))*(180./pi)
                times['calcAngle']=time.time()-start
                
                start=time.time()
                #rotate item using calculated axis and angle
                #axis=self.mapToParent(vstack(axis))
                self.rotate(angle=angle,x=axis[0],y=axis[1],z=axis[2], local=True)
                times['rotate']=time.time()-start
            
            
            start=time.time()
            #scale item to length of inputed vector
            self._resize(pvector)
            times['scl']=time.time()-start
            
            #update current vector
            self.vector=pvector
            return times
            
        

        


    def _resize(self, vector):
        #calculate current length, new length, and scaleFactor
        lnew=linalg.norm(vector)
        lcur=linalg.norm(self.vector)
        #scale using helper functions based on whether arrow length increases or decreases
        if lnew==0:
            self._zerocase=self.vector
            self.vector=array([0,0,0])
            self.hide()
            return
        elif lcur==0:
            self.show()
            self.vector=self._zerocase
        if lnew<lcur:
            self._scaleDown(vector)
        elif lnew>lcur:
            self._scaleUp(vector)
        else:
            return
       
      
    def _scaleUp(self, vector):
        #calculate z scale
        [dz,dx,dy]=[linalg.norm(vector)/linalg.norm(self.vector),1,1]
        #if current magnitude and desired magnitude are both shorter than unit vector
        if linalg.norm(self.vector)<1 and linalg.norm(vector)<1:
            #scale factor for x/y is same as z
            [dx,dy]=[dz,dz]
        #if both are longer than unit vector
        elif linalg.norm(self.vector)>1 and linalg.norm(vector)>1:
            #leave scale factor the same (presumably 1)
            [dx,dy]=[1,1]
        #presume that the arrow's magnitude changes from shorter than unit vector, to longer than
        #unit vector
        else:
            #only scale x/y up to unit vector configuration
            [dx,dy]=[linalg.norm(self.vector)**-1, linalg.norm(self.vector)**-1]
        
        #scale using superclass function
        self.scale(dx, dy, dz)
        
    def _scaleDown(self, vector):
        #calculate z scale
        [dz,dx,dy]=[linalg.norm(vector)/linalg.norm(self.vector),1,1]
        #if current magnitude and desired magnitude are both shorter than unit vector
        if linalg.norm(self.vector)<1 and linalg.norm(vector)<1:
            #scale factor for x/y is same as z
            [dx,dy]=[dz,dz]
        #if both are longer than unit vector
        elif linalg.norm(self.vector)>1 and linalg.norm(vector)>1:
            #leave scale factor the same (presumably 1)
            [dx,dy]=[1,1]
        #presume that the arrow's magnitude changes from longer than unit vector, to shorter
        #than unit vector
        else:
            #scale x/y from 1 down to scale of resulting vector
            [dx,dy]=[linalg.norm(vector), linalg.norm(vector)]
        
        #scale using superclass function
        self.scale( dx, dy, dz)
        
    def getVector(self):
         return self.vector
    def getPoint(self):
        return self.point
        
        
        
        
