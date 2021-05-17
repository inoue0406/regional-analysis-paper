import numpy as np

from scipy.ndimage.interpolation import rotate,zoom

# Custom Pytorch Data Augmentation for Video Data

class RandomRotateVideo(object):
    """RandomRotate for Video Data

    Args:
        degrees (float): Rotation Angle
    """
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, sample):
        past, future = sample['past'], sample['future']
        degree = (np.random.rand()-0.5)*self.degrees*2
        #print("random rotate: degrees=",degree)
        past= rotate(past,degree,axes=(2,3),reshape=False)
        future= rotate(future,degree,axes=(2,3),reshape=False)
        return {'past': past, 'future': future}


class RandomResizeVideo(object):
    """RandomRotate for Video Data

    Args:
        degrees (float): Rotation Angle
    """
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        past, future = sample['past'], sample['future']
        size1,size2 = past.shape[2],past.shape[3]
        z1 = 1 + np.random.rand()*self.factor
        z2 = 1 + np.random.rand()*self.factor
        #print("random resize z1,z2=",z1,z2)
        past= zoom(past,zoom=[1,1,z1,z2])
        future= zoom(future,zoom=[1,1,z1,z2])
        past= past[:,:,0:size1,0:size2]
        future= future[:,:,0:size1,0:size2]
        return {'past': past, 'future': future}
    
