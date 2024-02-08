import torch 

class multivector_collection:
    def __init__(self, tensor, metric):
        self.tensor = tensor
        self.metric = metric
        self.N = tensor.shape[0]
        self.d = len(metric)
        self.CGA_metric = [1]+metric+[-1]

    @property
    def CGA(self):
        #create empty array
        CGA_vertices = torch.zeros((self.N, self.d+2))

        #calculate innerproducts
        norm = torch.einsum('ni,ni,i->n', self.tensor, self.tensor, torch.Tensor(self.metric))

        #fill tensor
        CGA_vertices[:, 1:self.d+1] = self.tensor
        CGA_vertices[:, 0] = 0.5*(1-norm)
        CGA_vertices[:, -1] = 0.5*(1+norm)
        self.CGA_repr = CGA_vertices
        return 4*CGA_vertices

    def CGA_to_standard(self):
        factor = (self.CGA[:,0]+self.CGA[:,-1])
        return self.CGA[:,1:-1]/factor.unsqueeze(1)
        
    def translate(self, T):
        pass
    def dilate(self, D):
        pass
    def rotate(self, plane, theta):
        pass
    def sct(self, S):
        pass
