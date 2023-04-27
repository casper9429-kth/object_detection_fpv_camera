import numpy as np
from numpy.fft import fft2, ifft2, fftshift

def gaussfft(pic, t):
    
    ftransform = np.fft.fft2(pic)
    # Generate the power spectrum in centered frequency coordinates
    #(note that the factor (pi/umax) corresponds to (2*pi/usize))
    [usize, vsize] = np.shape(ftransform)
    t = t
    umax = int(usize/2)
    vmax = int(vsize/2)
    [u, v] = np.meshgrid(range(umax - usize, umax), range(vmax - vsize, vmax))
    # create gaussian distribution
    gaussian = (1/(2*(np.pi)*(t)))* np.exp(-(u**2+v**2)/(2*(t)))
    
    gaussian = fftshift(gaussian) # might not be needed
    
    # take forurier transform
    gaussian = fft2(gaussian)   
    
    # mult by picture
    ftransform_gauss = ftransform*gaussian
    
    return np.real(np.fft.ifft2(ftransform_gauss))




