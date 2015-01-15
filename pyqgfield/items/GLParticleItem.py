from pyqtgraph.opengl import GLMeshItem, MeshData
from numpy import *
from numpy.random import random

class GLParticleItem(GLMeshItem):
    def __init__(self, **kwds):
        """
        ============== =====================================================
        Arguments
        charge          this particle's electric charge  
        location        numpy.array([x,y,z]) indicating the x,y,z coordinates of this 
                        instance in the parent item's coordinate system
        ============== =====================================================
        """
        
        
        GLMeshItem.__init__(
            self,
            meshdata=MeshData.sphere(rows=10, cols=20, radius=.25),
            shader='shaded'
            )
        self.setColor(array([1.,1.,1.,1.]))
        
        ##load kwds into _opts
        self._opts={
            'charge': 0,
            'maxcharge': 5,
            'location': array([0.,0.,0.])
        }
        
        
        #instantiate instance to values from _opts
        self.setData(**kwds)
        
    #public function for setting particle attributes
    def setData(self, **kwds):
        if 'charge' in kwds and (kwds['charge'] is not self._opts['charge']):
                self._setCharge(kwds['charge'])
        if 'random' in kwds:
            if kwds['random']:
            #if random exists in kwds, place particle randomly in the interval 
            #x=[-random,random], y=[-random,random], z=[0,random]
                num=kwds.pop('random')
                self._setLocation([num-2*num*random(),num-2*num*random(),num*random()])
                return
        if 'location' in kwds and (kwds['location'] is not self._opts['location']):
            self._setLocation(kwds['location'])
        
            
    #helper function for updating particle charge
    def _setCharge(self, charge):
        self._opts['charge']=charge
        if charge>0:
            self.setColor(array([1,0,0,.5+.5*charge/self._opts['maxcharge']]))
        elif charge<0:
            self.setColor(array([0,0,1,.5+.5*charge/self._opts['maxcharge']]))
        else:
            self.setColor(array([1,1,1,.5]))
        
    
    #helper function for updating particle location
    def _setLocation(self, location):
        cur=self._opts['location']
        [dx,dy,dz]=[location[0]-cur[0],location[1]-cur[1], location[2]-cur[2]]
        self._opts['location']=location
        self.translate(dx,dy,dz,local=False)
        
        
    #public function for retrieving particle attributes
    def get(self, opt):
        if opt in self._opts:
            return(self._opts[opt])