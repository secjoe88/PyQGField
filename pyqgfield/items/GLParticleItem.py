from pyqtgraph.opengl import GLMeshItem, MeshData
from numpy import *

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
        self.opts={
            'charge':0.
            'location':array([0.,0.,0.])
        }
        GLMeshItem.__init__(
            self,
            meshdata=MeshData.sphere(rows=10, cols=20, radius=1),
            color=(1.,1.,1.,.75),
            shader='shaded'
            )
        self.setData(**kwds)
        
    #public function for setting particle attributes
    def setData(self, **kwds):
        if 'charge' in kwds and (kwds['charge'] not self.opts['charge']):
                self._setCharge(kwds['charge'])
        if 'location' in kwds and (kwds['location'] not self.opts['location']):
            self._setLocation(kwds['location'])
            
            
    #helper function for updating particle charge
    def _setCharge(self, charge):
        self.opts['charge']=charge
    
    #helper function for updating particle location
    def _setLocation(self, location)
        cur=self.opts['location']
        [dx,dy,dz]=[location[0]-cur[0],location[1]-cur[1], location[2]-cur[2]]
        self.translate(dx,dy,dz,local=False)
        
    #public function for retrieving particle attributes
    def get(self, opt):
        if opt in self.opts:
            return(self.opts[opt])