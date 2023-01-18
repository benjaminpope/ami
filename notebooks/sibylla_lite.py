import numpy as onp
import matplotlib.pyplot as plt

import jax.numpy as np 
from jax import grad, jit, vmap

from astropy.table import Table

from zodiax import Base # use this for object oriented Jax

mas2rad = np.pi / 180.0 / 3600.0/ 1000.0


def compute_DFTM1(x,y,uv,wavel):
    '''Compute a direct Fourier transform matrix, from coordinates x and y (milliarcsec) to uv (metres) at a given wavelength wavel.
    
    Parameters
    ----------
    x : 1D array
        x coordinates in milliarcsec
    y : 1D array
        y coordinates in milliarcsec
    uv : 2D array
        Baselines in metres
    wavel : float
        Wavelength in metres
        
    Returns
    -------
    dftm : 2D array
        DFT matrix
    '''

    # Convert to radians
    x = x * np.pi / 180.0 / 3600.0/ 1000.0
    y = y * np.pi / 180.0 / 3600.0/ 1000.0

    # get uv in nondimensional units
    uv = uv / wavel

    # Compute the matrix
    dftm = np.exp(-2j* np.pi* (np.outer(uv[:,0],x)+np.outer(uv[:,1],y)))

    return dftm

def apply_DFTM1(image,dftm):
    '''Apply a direct Fourier transform matrix to an image.
    
    Parameters
    ----------
    image : 2D array
        Image to be transformed
    dftm : 2D array
        DFT matrix

    Returns
    -------
    FT : 2D array
        Fourier transform of the image

    '''
    image /= image.sum()
    return np.dot(dftm,image.ravel())


def compute_DFTM2(input_coordinates, output_coordinates, axis=0):
    '''
    Compute a 2-sided direct Fourier transform matrix, from coordinates x and y (milliarcsec) 
    to uv (metres) at a given wavelength wavel.

    Parameters
    ----------
    steps_in : tuple
        (x_step, y_step) in milliarcsec
    array_sizes : tuple
        (x_size, y_size) in pixels
    wavel : float
        Wavelength in metres
    pscale : float
        Pixel scale in milliarcsec
    axis : int
        Axis to compute DFTM along

    Returns
    -------
    ftm : 2D array
        One axis DFT matrix
    '''

    input_to_output = np.outer(input_coordinates, output_coordinates)

    ftm = np.exp(2. * np.pi * 1j * input_to_output)

    if axis != 0:
        return ftm
    else:
        return ftm.T

def both_DFTM2(u, v, img_size, wavel, pscale):
    '''Compute a 2-sided direct Fourier transform matrix, from coordinates x and y (milliarcsec) 
    to uv (metres) at a given wavelength wavel.
    
    Parameters
    ----------
    u: 1D array
        u coordinates of the baselines in metres
    v: 1D array
        v coordinates of the baselines in metres
    img_size : int
        Image size in pixels
    pscale : float
        Pixel scale in mas/pixel
    wavel : float
        Wavelength in metres
    
    Returns
    -------
    LL, RR : 2D arrays
        Left and Right DFT matrices
    '''

    # Compute the matrix
    output_coordinates = u/wavel

    input_coordinates = (np.arange(img_size) - img_size/2)*pscale*mas2rad

    LL = compute_DFTM2(input_coordinates, output_coordinates, axis=0)
    
    # Compute the other matrix
    output_coordinates = v/wavel
    input_coordinates = (np.arange(img_size) - img_size/2)*pscale*mas2rad

    RR = compute_DFTM2(input_coordinates, output_coordinates, axis=1)
    return LL, RR


def apply_dftm2(image,u, v, pscale, wavel):
    '''Apply a 2-sided direct Fourier transform matrix to an image.    

    Parameters
    ----------
    image : 2D array
        Image to be transformed
    uv : 2D array
        Baselines in metres
    pscale : float
        Pixel scale in mas/pixel
    wavel : float
        Wavelength in metres

    Returns
    -------
    FT : 2D array
        Fourier transform of the image
    '''

    image /= np.sum(image)

    img_size = image.shape[0]

    LL, RR = both_DFTM2(u, v, img_size, wavel, pscale)
    
    FT = np.dot(LL, np.dot(image, RR))

    return FT

###-----------------###

Array = np.ndarray
from equinox import static_field

class UVGrid(Base):
    '''
    A Zodiax layer to do a 2D Fourier transform of an image onto a uniformly-sampled uv plane grid 
    using a 2-sided matrix Fourier transform.
    '''
    u: Array
    v: Array
    pscale: float
    wavel: float
    img_size: int
    LL: Array
    RR: Array

    def __init__(self, u, v, pscale, wavel, img_size):
        '''
        Parameters
        ----------
        u : 1D array
            u coordinates of the baselines in metres
        v : 1D array
            v coordinates of the baselines in metres
        pscale : float
            Pixel scale in mas/pixel
        wavel : float
            Wavelength in metres
        img_size : int
            Image size in pixels
        '''
        self.u = u
        self.v = v
        self.pscale = pscale
        self.wavel = wavel
        self.img_size = img_size

        self.LL, self.RR = both_DFTM2(u, v, img_size, wavel, pscale)

    def __call__(self,image):
        '''
        Apply a 2-sided direct Fourier transform matrix to an image.

        Parameters
        ----------
        image : 2D array
            Image to be transformed

        Returns
        -------
        FT : 2D array
            Fourier transform of the image
        '''

        image /= np.sum(image)

        FT = np.dot(self.LL, np.dot(image, self.RR))
        return FT

class UVNonUniform(Base):
    '''
    A Zodiax layer to do a 1D Fourier transform of an image onto an arbitrary uv array 
    using a 1-sided matrix Fourier transform. Output will not by default be image-shaped.
    '''
    
    uv: Array
    x: Array
    y: Array
    wavel: float
    DFTM: Array

    def __init__(self, uv, x, y, wavel):
        '''
        Parameters
        ----------
        uv : 2D array
            uv coordinates of the baselines in metres
        x : 1D array
            x coordinates of the image in milliarcsec
        y : 1D array
            y coordinates of the image in milliarcsec
        wavel : float
            Wavelength in metres

        '''
        self.uv = uv
        self.x = x
        self.y = y
        self.wavel = wavel
        
        self.DFTM = compute_DFTM1(self.x,self.y,uv,self.wavel)

    def __call__(self,image):
        '''
        Apply a 2-sided direct Fourier transform matrix to an image.

        Parameters
        ----------
        image : 2D array
            Image to be transformed

        Returns
        -------
        FT : 2D array
            Fourier transform of the image
        '''

        image /= np.sum(image)

        FT = np.dot(self.DFTM, image.ravel())#.reshape(self.u.shape[0], self.uv.shape[1])
        return FT

def makebaselines(mask):
    """
    ctrs_eqt (nh,2) in m
    returns np arrays of eg 21 baselinenames ('0_1',...), eg (21,2) baselinevectors (2-floats)
    in the same numbering as implaneia
    """
    nholes = mask.shape[0]
    blist = []
    for i in range(nholes):
        for j in range(nholes):
            if i < j:
                blist.append((i, j))
    barray = onp.array(blist).astype(onp.int)
    # blname = []
    bllist = []
    for basepair in blist:
        # blname.append("{0:d}_{1:d}".format(basepair[0],basepair[1]))
        baseline = mask[basepair[0]] - mask[basepair[1]]
        bllist.append(baseline)
    return barray, np.array(bllist)

def maketriples_all(mask,verbose=False):
    """ returns int array of triple hole indices (0-based), 
        and float array of two uv vectors in all triangles
    """
    nholes = mask.shape[0]
    tlist = []
    for i in range(nholes):
        for j in range(nholes):
            for k in range(nholes):
                if i < j and j < k:
                    tlist.append((i, j, k))
    tarray = onp.array(tlist).astype(onp.int)
    if verbose:
        print("tarray", tarray.shape, "\n", tarray)

    tname = []
    uvlist = []
    # foreach row of 3 elts...
    for triple in tarray:
        tname.append("{0:d}_{1:d}_{2:d}".format(
            triple[0], triple[1], triple[2]))
        if verbose:
            print('triple:', triple, tname[-1])
        uvlist.append((mask[triple[0]] - mask[triple[1]],
                       mask[triple[1]] - mask[triple[2]]))
    # print(len(uvlist), "uvlist", uvlist)
    if verbose:
        print(tarray.shape, onp.array(uvlist).shape)
    return tarray, np.array(uvlist)


# def makebaselines_fast(mask):
#     '''
#     Generate a list of baselines from a mask of holes, removing the zero baseline, redundant baselines, and going only forwards 
#     and not backwards in the list of baselines. This is a faster version of makebaselines.
#     '''
#     thisu = onp.subtract.outer(mask[:,0],mask[:,0])
#     thisu = thisu.T[np.where(~np.eye(thisu.shape[0],dtype=bool))]

#     thisv = onp.subtract.outer(mask[:,1],mask[:,1])
#     thisv = thisv.T[np.where(~np.eye(thisv.shape[0],dtype=bool))]

#     # remove baselines which are minus another baseline
#     thisu = thisu[np.where(thisu >= 0)]
#     thisv = thisv[np.where(thisv >= 0)]
