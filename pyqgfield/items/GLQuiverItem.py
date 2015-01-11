from pyqgfield.items import GLArrowItem
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from numpy import *

__all__=['GLQuiverItem']
class GLQuiverItem(GLGraphicsItem):
    def __init__(self, **kwds):
        """
        ============== =====================================================
        Arguments
        points             (N,3) array points[i]=[xi,yi,zi] specifying where to place
                        vectors
        vectors            (N,3) array vectors[i]=[vx_i,vy_i,vz_i] specifying the
                        vector placed at pts[i]
        ============== =====================================================
        """
        self.arrows=None
        GLGraphicsItem.__init__(self)
        self.setData(**kwds)
        
        
        
    def setData(self, **kwds):
        if len(kwds)==0:
            return
        try:
            points=array(kwds.pop('points'))
            vectors=array(kwds.pop('vectors'))
        except KeyError:
            print('Invalid Key')
            return
        if (not shape(points)[1]==3) or (not shape(points)==shape(vectors)):
            print('Invalid vector dimensions. Points and Vectors must be (N,3) in shape')
            return
        
        #update point and vector of each arrow in self.arrows
        if (self.arrows is None) or (not len(self.arrows)==len(points)):
            self.arrows=[GLArrowItem() for i in range(len(points))]
            for arrow in self.arrows:
                arrow.setParentItem(self)
        i=0
        for row in points:
            self.arrows[i].updateData(point=points[i],vector=vectors[i])
            i+=1
            
        self.update()
            
    

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
                
            
    
            
