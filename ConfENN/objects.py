import torch 
from CGENN.algebra.cliffordalgebra import *

class multivector_collection:
    def __init__(self, tensor, metric):
        self.tensor = tensor
        self.metric = metric
        self.N = tensor.shape[0]
        self.d = len(metric)
        self.CGA_metric = [1]+metric+[-1]
        self.CA= CliffordAlgebra(metric)
        self.ConfCA = CliffordAlgebra([1]+metric+[-1])

    @property
    def CGA(self):
        #create empty array
        CGA_vertices = torch.zeros((self.N, self.d+2))

        #calculate innerproducts
        norm = torch.einsum('ni,ni,i->n', self.tensor, self.tensor, torch.Tensor(self.metric))

        #fill tensor
        CGA_vertices[:, 1:self.d+1] = self.tensor
        CGA_vertices[:, 0] = (norm-1)/2
        CGA_vertices[:, -1] = -(norm+1)/2
        CGA_repr = self.ConfCA.embed_grade(CGA_vertices,1)
        return CGA_repr
    @property
    def n(self):
        return self.ConfCA.embed_grade(torch.tensor([1]+[0]*self.d+[1]),1)

    @property
    def n_bar(self):
        return self.ConfCA.embed_grade(torch.tensor([1]+[0]*self.d+[-1]),1)

    @property
    def e(self):
        return self.ConfCA.embed_grade(torch.tensor([1]+[0]*self.d+[0]),1)
    
    @property
    def e_bar(self):
        return self.ConfCA.embed_grade(torch.tensor([0]+[0]*self.d+[1]),1)

    def CGA_to_standard(self, mvs):
        mvs = self.ConfCA.get_grade(mvs, 1)
        factor = (mvs[:,0]+mvs[:,-1])/-2
        return mvs[:,1:-1]/factor.unsqueeze(1)
        
    def apply_rotor(self, rotor, mv):
        left = self.ConfCA.geometric_product(rotor, mv)
        right = self.ConfCA.geometric_product(left, self.ConfCA.inverse(rotor))
        return right


    #Conformal transformation in Euclidean space
    def translate_std(self, T):
        self.tensor += T
    def dilate_std(self, alpha):
        self.tensor *= torch.exp(alpha)
    def rotate_std(self, v1, v2, theta):
        pass

    def sct_std(self, S):
        pass

    #Conformal transformations in CGA
    def translate(self, T):
        translation = self.ConfCA.embed_grade(T, 1)
        rotor = self.ConfCA.geometric_product(self.n, translation)/2
        rotor[..., 0] += 1
        translated = self.apply_rotor(rotor, self.CGA)
        self.tensor = self.CGA_to_standard(translated)
    
    def dilate(self, alpha):
        N = self.ConfCA.geometric_product(self.e, self.e_bar)
        rotor = torch.sinh(alpha/2)*N
        rotor[..., 0] += torch.cosh(alpha/2)
        translated = self.apply_rotor(rotor, self.CGA)
        self.tensor = self.CGA_to_standard(translated)
        
    def rotate(self, plane_1, plane_2, theta):
        plane_1 = self.ConfCA.embed_grade(plane_1, 1)
        plane_2 = self.ConfCA.embed_grade(plane_2,1)
        rotor = self.ConfCA.geometric_product(plane_1, plane_2)*torch.sin(theta)
        rotor[...,0] = torch.cos(theta)
        rotated = self.apply_rotor(rotor, self.CGA)
        self.tensor = self.CGA_to_standard(rotated)

    def invert(self):
        return - self.apply_rotor(self.e, self.CGA)

    def sct(self, S):
        #invert
        inverted = self.invert()
        #translate
        translate = self.translate(S)
        #invert
        return self.invert()
        
