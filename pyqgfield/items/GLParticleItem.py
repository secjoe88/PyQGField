from pyqtgraph.opengl import GLMeshItem, MeshData
from numpy import *

class GLParticleItem(GLMeshItem):
    def __init__(self, **kwds):
        """
        ============== =====================================================
        Arguments
        meshdata       MeshData object from which to determine geometry for 
                       this item.
        color          Default face color used if no vertex or face colors 
                       are specified.
        edgeColor      Default edge color to use if no edge colors are
                       specified in the mesh data.
        drawEdges      If True, a wireframe mesh will be drawn. 
                       (default=False)
        drawFaces      If True, mesh faces are drawn. (default=True)
        shader         Name of shader program to use when drawing faces.
                       (None for no shader)
        smooth         If True, normal vectors are computed for each vertex
                       and interpolated within each face.
        computeNormals If False, then computation of normal vectors is 
                       disabled. This can provide a performance boost for 
                       meshes that do not make use of normals.
        ============== =====================================================
        """
        GLMeshItem.__init__(
            self,
            meshdata=MeshData.sphere(rows=10, cols=20, radius=.25),
            color=(0,0,1,.75),
            shader='shaded'
            )
        
