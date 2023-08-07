import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
from skimage.transform import rescale




class Sampling:
    def __init__(self,X,C):
        self.X = X
        self.C = C
    def D3CASSIsampling(self):
        [m, n, l] = np.shape(self.X)
        Y = np.zeros((m, n))
        for i in range(l):
            # Load ith multispectral image
            X1 = self.X[:, :, i]
            # Load ith coded aperture
            C1 = self.C==(i+1)
            # To compute the compressive measurement without prism dispersion
            Y = Y + X1*C1
        return Y

class Patterns:
    def __init__(self,N):
        self.N = N
    def Uniform(self):
        N = self.N
        L = 2
        M = int(self.N/(L+1))+1
        #K = np.array([[1, 3, 2], [2, 1, 3], [3, 2, 1]])
        #K = np.array([[1, 2, 3], [3, 1, 2], [2, 3, 1]])
        #K = np.array([[1, 3, 2], [2, 1, 3], [3, 2, 1]])
        K = np.array([[2, 1, 3], [3, 2, 1], [1, 3, 2]])
        G = np.kron(np.ones((M,M)), K)
        G = G[0:N,0:N]
        return G
    def BayerFilterBGGR(self):
        N = self.N
        L = 2
        M = int(self.N/(L))
        K = np.array([[1, 2], [2, 3]])
        #K = np.array([[2, 1], [3, 2]])
        #K = np.array([[2, 3], [1, 2]])
        #K = np.array([[3, 2], [2, 1]])
        G = np.kron(np.ones((M,M)), K)
        G = G[0:N,0:N]
        #K = np.array([[2, 1, 3, 1], [3, 2, 1, 3], [1, 3, 2, 1], [3, 2, 1, 3]])
        #G = np.random.randint(3, size=(N, N))+1
        #C = np.zeros((N,N,3))
        #for i in range(3):
        #    C[:,:,i] = G == i 
        return G

class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        N = sidelength
        P = Patterns(N)
        G = P.Uniform()
        img = get_mosaic_tensor(sidelength,G)
        r = img[0,:,:].view(-1, 1)
        self.pixels = r

        y = get_mgridMosaic(sidelength, 2)
        self.coords = y
        #print(self.coords[15])
        #plt.imshow(img[0,:,:])
        #plt.show()

    def __len__(self):
        return 1#len(self.coords)

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels


def get_mgridMosaic(N, dim=2):
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    mgrid1 = np.zeros((N*N,dim+1),dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    #fig, axes = plt.subplots(1,2, figsize=(18,6))
    #axes[0].imshow(xv)
    #axes[1].imshow(yv)
    #plt.show()
    P = Patterns(N)
    G = P.Uniform()
    G = G-2.0
    xv1 = np.float32(xv.reshape(-1, 1))
    yv1 = np.float32(yv.reshape(-1, 1))
    G1 = np.float32(G.reshape(-1, 1))
    mgrid1[:,0] = (xv1[:,0])
    mgrid1[:,1] = (yv1[:,0])
    mgrid1[:,2] = G1[:,0]
    mgrid = torch.from_numpy(mgrid1)
    return mgrid 

def get_mgridRGB(N, dim=2):
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    mgrid1 = np.zeros((N*N,dim+1),dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    #fig, axes = plt.subplots(1,2, figsize=(18,6))
    #axes[0].imshow(xv)
    #axes[1].imshow(yv)
    #plt.show()
    P = Patterns(N) # To Create a pattern object
    G = P.Uniform() # To compute Bayer Filter
    G = G-2.0
    xv1 = np.float32(xv.reshape(-1, 1))
    yv1 = np.float32(yv.reshape(-1, 1))
    G1 = np.float32(G.reshape(-1, 1))
    mgrid1[:,0] = (xv1[:,0])
    mgrid1[:,1] = (yv1[:,0]) 
    mgrid1[:,2] = G1[:,0]

    a = np.concatenate((mgrid1[:,0], mgrid1[:,0],mgrid1[:,0]))
    b = np.concatenate((mgrid1[:,1], mgrid1[:,1],mgrid1[:,1]))
    R = mgrid1[:,2]*0
    c = np.concatenate((R+1, R, R-1))
    mgrid2 = np.vstack((a,b,c))
    mgrid2 = mgrid2.transpose()
    mgrid = torch.from_numpy(mgrid2)
    return mgrid 


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def get_mosaic_tensor(N,G):
    img = Image.fromarray(skimage.data.camera())  
    temp = skimage.data.camera()
    #img1 = Image.fromarray(skimage.data.astronaut(),mode="RGB")
    rgb = skimage.data.astronaut()
    rgb = (rgb/255).astype(np.float32)

    #im = utils.normalize(plt.imread('data\parrot.png').astype(np.float32), True)
    #im = utils.normalize(plt.imread('data\kodim23.png').astype(np.float32), True)
    #im = cv2.resize(im, None, fx=1/8, fy=1/8, interpolation=cv2.INTER_AREA) 
    #rgb = im[0:N,0:N,:]

    [a,b,c]=np.shape(rgb)
    tp = np.min([a,b])
    rgb = rgb[0:tp,0:tp,:]
    temp = np.zeros((N,N,3))
    BM = np.size(rgb,0)
    temp[:,:,0] = rescale(rgb[:,:,0], N/BM, anti_aliasing=True)*255
    temp[:,:,1] = rescale(rgb[:,:,1], N/BM, anti_aliasing=True)*255
    temp[:,:,2] = rescale(rgb[:,:,2], N/BM, anti_aliasing=True)*255
    X = (np.rint(temp)).astype(np.uint8)
    #plt.imshow(G)
    #plt.show()
    S = Sampling(X,G)
    Y = S.D3CASSIsampling()
    Y = (np.rint(Y)).astype(np.uint8)
  
    img1 = Image.fromarray(Y)
  
    transform = Compose([
        Resize(N),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    img1 = transform(img1)
    return img1