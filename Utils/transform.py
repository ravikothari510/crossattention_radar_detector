from matplotlib.pyplot import axis
import numpy as np
from scipy import interpolate
from torchvision import transforms
import torch
import random


def data_transform():
    rad = {}
    ra_transform = transforms.Compose([
    transforms.Normalize(
        mean = [678.5],
        std = [1352.6]
    )])

    rd_transform = transforms.Compose([
        transforms.Normalize(
            mean = [34.1],
            std = [2.93]
        )])
             
    ad_transform = transforms.Compose([
        transforms.Normalize(
            mean = [35.3],
            std = [2.07]
        )])

    rad['ra_map'] = ra_transform
    rad['rd_map'] = rd_transform
    rad['ad_map'] = ad_transform
    return rad



def polToCart(map):
    y = np.linspace(50,0,256)
    x = np.arcsin(np.linspace(-1,1,256))
    z = map
    f = interpolate.interp2d(x,y,z,kind='cubic')
    x_cart = np.linspace(-25,25,256)
    y_cart = np.linspace(0,50,256)
    xx,yy = np.meshgrid(x_cart,y_cart)
    r_cart = np.sqrt(xx**2 + yy**2)
    a_cart = np.arctan(xx/(yy+1e-10))
    cart = np.zeros((256,256))
    for i in range(256):
        for j in range(256):
            cart[i,j] = f(a_cart[i,j],r_cart[i,j])

    return np.flip(cart,axis=0)

def get_index(x,y):
    x_cart = np.linspace(-25,25,256)
    y_cart = np.linspace(0,50,256)

    x_ind = np.argmin(np.abs(x-x_cart))
    y_ind = np.argmin(np.abs(y-y_cart))
    return x_ind, y_ind

def cartToPol(x,y):
    r = np.sqrt(x**2+y**2)
    a = np.arctan(x/(y+1e-10))
    return r,a
    
def get_index_pol(r,a):
    range_array = np.linspace(50,0,num=256)
    angle_array = np.arcsin(np.linspace(-1,1,num=256))
    r_ind = np.argmin(np.abs(r-range_array))
    a_ind = np.argmin(np.abs(a-angle_array))

    return r_ind,a_ind
def pol2cart(r,a):
    r = (50.0/255)*(255-r)
    theta = -1 + ((2.0/256)*a)
    y = r*np.cos(theta)
    x = r*np.sin(theta)
    return get_index(x,y)

def noise_add(r_map, device):

    if random.choice([0,1]) and r_map != 0:
        noise_map = torch.normal(0, 0.1, size=r_map.shape).to(device=device)
        return r_map+noise_map
    else:
        return r_map