import numpy as onp
import matplotlib.pyplot as plt

import jax.numpy as np 
from jax import grad, jit, vmap
import jax

'''-----------------
Visibility Tools

A set of functions to compute visibilities from a geometric models.

-----------------'''


mas2rad = np.pi / 180.0 / 3600.0/ 1000.0 # convert milliarcsec to radians


def j1(x):
    '''
    Compute the first order Bessel function of the first kind.

    Parameters
    ----------
    x : float
        Input value

    Returns
    -------
    float
        Value of the Bessel function

    '''
    return jax.scipy.special.bessel_jn(x,v=1,n_iter=50)[1]

def j0(x):
    '''
    Compute the zeroth order Bessel function of the first kind.

    Parameters
    ----------
    x : float
        Input value

    Returns
    -------
    float
        Value of the Bessel function
        
    '''

    dummy = jax.scipy.special.bessel_jn(x,v=0,n_iter=50)[0]
    return jax.lax.select(~np.isfinite(dummy), 1., dummy)

@jit
@vmap
def jinc(x):
    '''
    Compute the Jinc function.

    Parameters
    ---------- 
    x : float
        Input value

    Returns
    -------
    float
        Value of the Jinc function

    '''
    dummy = 2*(j1(x)/x)
    return jax.lax.select(~np.isfinite(dummy), 1., dummy)

def vis_gauss(d,u,v):
    '''
    Compute the visibility of a Gaussian source.

    Parameters
    ----------
    d : float
        Diameter of the Gaussian source in milliarcsec
    u : float
        u coordinate of the baseline in wavelengths
    v : float
        v coordinate of the baseline in wavelengths

    Returns
    -------
    float
        Visibility of the Gaussian source

    '''
    d = mas2rad*d
    return np.exp(-(np.pi*d*np.sqrt(u**2+v**2))**2/4./np.log(2))

def vis_ud(d,u,v):
    '''
    Compute the visibility of a uniform disk source.

    Parameters
    ----------
    d : float
        Diameter of the uniform disk source in milliarcsec
    u : float
        u coordinate of the baseline in wavelengths
    v : float
        v coordinate of the baseline in wavelengths

    Returns
    -------
    float
        Visibility of the uniform disk source

    '''
    r = np.sqrt(u**2+v**2)
    diam = d*mas2rad
    t = jinc(np.pi*diam*r)
    return t

def vis_ring(d,u,v):
    '''
    Compute the visibility of a ring source.

    Parameters
    ---------- 
    d : float
        Diameter of the ring source in milliarcsec
    u : float
        u coordinate of the baseline in wavelengths
    v : float
        v coordinate of the baseline in wavelengths

    Returns
    -------
    float
        Visibility of the ring source

    '''
    r = np.sqrt(u**2+v**2)
    diam = d*mas2rad
    t = j0(np.pi*diam*r)
    return t
    
def vis_ellipse_disk(semi_axis,ecc,theta,u,v):
    '''
    Compute the visibility of an elliptical disk source.

    Parameters
    ----------
    semi_axis : float
        Semi-major axis of the elliptical disk source in milliarcsec
    ecc : float
        Eccentricity of the elliptical disk source
    theta : float
        Position angle of the elliptical disk source in degrees
    u : float
        u coordinate of the baseline in wavelengths
    v : float
        v coordinate of the baseline in wavelengths

    Returns
    -------
    float
        Visibility of the elliptical disk source
    '''
    
    semi_axis = semi_axis
    thetad = np.pi*theta/180.
    u1, v1 = u*np.cos(thetad)+v*np.sin(thetad), -u*np.sin(thetad)+v*np.cos(thetad)
    u1, v1 = u1, v1*np.sqrt(1-ecc**2.)
    
    return vis_ud(0.5*semi_axis,u1,v1)

def vis_ellipse_thin(semi_axis,ecc,theta,thick,u,v):
    '''
    Compute the visibility of an elliptical disk source with a thin ring.

    Parameters
    ----------
    semi_axis : float
        Semi-major axis of the elliptical disk source in milliarcsec
    ecc : float
        Eccentricity of the elliptical disk source
    theta : float
        Position angle of the elliptical disk source in degrees
    thick : float
        Thickness of the ring in milliarcsec
    u : float
        u coordinate of the baseline in wavelengths
    v : float
        v coordinate of the baseline in wavelengths
    
    Returns
    -------
    float
        Visibility of the elliptical disk source with a thin ring
    '''
    
    ad, bd = semi_axis, semi_axis*np.sqrt(1.-ecc**2.)
    a2, b2 = semi_axis-thick, (semi_axis-thick)*np.sqrt(1.-ecc**2)
    n1, n2 = ad*bd, a2*b2
    return vis_ellipse_disk(semi_axis,ecc,theta,u,v)-n2/n1*vis_ellipse_disk(semi_axis-thick,ecc,theta,u,v)

def vis_ellipse_gauss(semi_axis,thick,gausswidth,ecc,theta,u,v):
    '''
    Compute the visibility of an elliptical disk source with a Gaussian ring.
    
    Parameters
    ----------
    semi_axis : float
        Semi-major axis of the elliptical disk source in milliarcsec
    thick : float
        Thickness of the ring in milliarcsec
    gausswidth : float
        Width of the Gaussian ring in milliarcsec
    ecc : float
        Eccentricity of the elliptical disk source
    theta : float
        Position angle of the elliptical disk source in degrees
    u : float
        u coordinate of the baseline in wavelengths
    v : float
        v coordinate of the baseline in wavelengths
    
    Returns
    -------
    float
        Visibility of the elliptical disk source with a Gaussian ring
    '''
    return vis_gauss(gausswidth,u,v)*vis_ellipse_thin(semi_axis,thick,ecc,theta,u,v)

def vis_ellipse_high_contrast(semi_axis,ecc,theta,thick,con,u,v):
    '''
    Compute the visibility of an elliptical disk source with a thin ring and high contrast.
    
    Parameters
    ----------
    semi_axis : float
        Semi-major axis of the elliptical disk source in milliarcsec
    ecc : float
        Eccentricity of the elliptical disk source
    theta : float
        Position angle of the elliptical disk source in degrees
    thick : float
        Thickness of the ring in milliarcsec
    con : float 
        Contrast of the ring
    u : float
        u coordinate of the baseline in wavelengths
    v : float
        v coordinate of the baseline in wavelengths

    Returns
    -------
    float
        Visibility of the elliptical disk source with a thin ring and high contrast
    '''
    l2 = 1. / (con + 1)
    l1 = 1 - l2
    
    vises = vis_ellipse_thin(semi_axis,ecc,theta,thick,u,v)
    norm = vis_ellipse_thin(semi_axis,ecc,theta,thick,np.array([1e-5]),np.array([1e-5]))
    vises = l2*(vises/norm) + l1
    return vises

def vis_ring_high_contrast(diam,con,u,v):

    '''
    Compute the visibility of a ring source with high contrast.
    
    Parameters
    ----------
    diam : float
        Diameter of the ring source in milliarcsec
    con : float
        Contrast of the ring
    u : float
        u coordinate of the baseline in wavelengths
    v : float
        v coordinate of the baseline in wavelengths

    Returns
    -------
    float
        Visibility of the ring source with high contrast
    '''

    l2 = 1. / (con + 1)
    l1 = 1 - l2
    
    vises = vis_ring(diam,u,v)
    vises = l2*(vises) + l1
    return vises