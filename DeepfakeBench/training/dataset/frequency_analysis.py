import numpy as np
from scipy.signal import convolve2d
import ipdb


class LGA():
    def __init__(self, kernel_size,sigma,constant):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.constant = constant
    def gaussian_kernel(self, size, sigma):
        
        ax = np.linspace(-(size // 2), size // 2, size)
        ay = np.linspace(-(size // 2), size // 2, size)
        xx, yy = np.meshgrid(ax, ay)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)  
        return kernel

    def get_lga(self,img_gray, kernel_size, sigma,constant):
        # ipdb.set_trace()
        Wx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        Wy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        Gx = convolve2d(img_gray, Wx, mode='same', boundary='wrap')
        Gy = convolve2d(img_gray, Wy, mode='same', boundary='wrap')
        gradient = np.sqrt(Gx**2 + Gy**2 + constant)
        Kernel = self.gaussian_kernel(kernel_size, sigma)
        auto_correlation = convolve2d(gradient, Kernel, mode='same', boundary='wrap')   
        lga = gradient - auto_correlation
        return lga

    def forward(self,img_gray):
        lga = self.get_lga(img_gray,self.kernel_size,self.sigma,self.constant)
        return lga

    def __call__(self, img_gray):
        return self.forward(img_gray)


class LVP():
    def __init__(self, weights):
        self.H, self.W = 224, 224
        self.weights = weights
    
    def calculate_Ei(self,Pc, Pn):
        if Pc - Pn > 0:
            return 1
        else:
            return 0
        
    def calculate_LVP(self, img_gray, i, j):
        Pc = img_gray[i, j]
        neighbors = [
            (i-1, j-1), (i-1, j), (i-1, j+1),  
            (i, j-1),                         (i, j+1),  
            (i+1, j-1), (i+1, j), (i+1, j+1)   
        ]
        LVP_value = 0
        for idx, (ni, nj) in enumerate(neighbors):
            if 0 <= ni < self.H and 0 <= nj < self.W:  
                Pn = img_gray[ni, nj]
                Ei_pn = self.calculate_Ei(Pc, Pn)
                LVP_value += Ei_pn * self.weights[idx]  
        return LVP_value

    def forward(self,img_gray):
        LVP_result = np.zeros_like(img_gray, dtype=float)
        for i in range(1,self.H-1):
            for j in range(1,self.W-1):
                LVP_result[i,j] = self.calculate_LVP(img_gray, i,j)
        return LVP_result
    
    def __call__(self, img_gray):
        return self.forward(img_gray)





# def calculate_Ei(Pc, Pn):
#     if Pc - Pn > 0:
#         return 1
#     else:
#         return 0
        
# def calculate_LVP(img_gray, i, j):
#     H, W = img_gray.shape
#     Pc = img_gray[i, j]
#     neighbors = [
#         (i-1, j-1), (i-1, j), (i-1, j+1),  
#         (i, j-1),                         (i, j+1),  
#         (i+1, j-1), (i+1, j), (i+1, j+1)   
#     ]
#     LVP_value = 0
#     for idx, (ni, nj) in enumerate(neighbors):
#         if 0 <= ni < H and 0 <= nj < W:  
#             Pn = img_gray[ni, nj]
#             Ei_pn = calculate_Ei(Pc, Pn)
#             LVP_value += Ei_pn * weights[idx]  
#     return LVP_value

# def apply_LVP(img_gray):
#     H, W = img_gray.shape
#     LVP_result = np.zeros_like(img_gray, dtype=float)
#     for i in range(1, H-1):
#         for j in range(1, W-1):
#             LVP_result[i, j] = calculate_LVP(img_gray, i, j)



