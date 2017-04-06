#+
# NAME:
#     rayleighsommerfeld
#
# PURPOSE:
#     Computes Rayleigh-Sommerfeld back-propagation of
#     a normalized hologram of the type measured by
#     digital video microscopy
#
# CATEGORY:
#     Holographic microscopy
#
# CALLING SEQUENCE:
#     b = rayleighsommerfeld(a, z)
#
# INPUTS:
#     a: hologram recorded as image data normalized
#         by a background image. 
#     z: displacement from the focal plane [pixels]
#         If z is an array of displacements, the field
#         is computed at each plane.
#
# KEYWORDS:
#     lamb: Wavelength of light in medium [micrometers]
#         Default: 0.632 -- HeNe in air
#     mpp: Micrometers per pixel
#         Default: 0.135
#
# KEYWORD_FLAGS:
#     nozphase: By default, a phase factor of kz is removed 
#         from the propagator to eliminate the (distracting)
#         axial phase gradient.  Setting this flag leaves 
#         this factor (and the axial phase ramp) in.
#     hanning: If set, use Hanning window to suppress
#         Gibbs phenomeon.
#
# OUTPUTS:
#     b: complex field at height z above image plane, 
#         computed by convolution with
#         the Rayleigh-Sommerfeld propagator.
#
# REFERENCES:
# 1. S. H. Lee and D. G. Grier, 
#   "Holographic microscopy of holographically trapped 
#   three-dimensional structures, "
#   Optics Express, 15, 1505-1512 (2007).
#
# 2. J. W. Goodman, "Introduction to Fourier Optics, "
#    (McGraw-Hill, New York 2005).
#
# 3. G. C. Sherman, 
#   "Application of the convolution theory to Rayleigh's integral
#   formulas, "
#   Journal of the Optical Society of America 57, 546-547 (1967).
#
# PROCEDURE:
#     Convolution with Rayleigh-Sommerfeld propagator using
#     Fourier convolution theorem.
#
# EXAMPLES:
# Visualize the intensity as a function of height z above the image
# plane given a normalized holographic image a.

# >>> lamb = 0.6328 / 1.336 # HeNe in water
# >>> for z in range(1., 200):
# >>>    tvscl, abs(rayleighsommerfeld(a, z, lamb=lamb))**2 & #$
#
# Visualize the phase singularity due to a particle located along the
# line y = 230 in the image a, assuming that the particle is no more
# than 200 pixels above the image plane.
#
# >>> help, a
# A     FLOAT     = Array[480, 640]
# >>> phi = fltarr(640, 200)
# >>> for z in range(1., 200): ##One line for loop
# >>> phiz = atan(rayleighsommerfeld(a, z, lamb=lamb), /phase) & #$
# >>> phi[z-1, :] = phiz[230, :]
# >>> tvscl, phi
#
# MODIFICATION HISTORY:
# 11/28/2006: Sang-Hyuk Lee, New York University, 
#   Original version (called FRESNELDHM.PRO).
#   Based on fresnel.pro by D. G. Grier
# 10/21/2008: David G. Grier, NYU.  Adapted from FRESNELDHM
#   to provide more uniform interface.
# 11/07/2009: DGG  Code and documentation clean-up.  Properly
#   handle places where qsq > 1.  Implemented NOZPHASE.
# 11/09/2009: DGG Initial implementation of simultaneous projection
#   to multiple z planes for more efficient volumetric
#   reconstructions.  Eliminate matrix reorganizations.
# 06/10/2010: DGG Documentation fixes.  Added COMPILE_OPT.
# 07/27/2010: DGG Added HANNING.
# 10/20/2010: DGG Subtract 1 from (normalized) hologram
#   rather than expecting user to do subtraction.
# 06/24/2012 DGG Overhauled computation of Hqz.
# 06/04/2017 Julius B Kirkegaard & Francois Peaudecerf: pyCUDA version 
#             massive speed gain (retuns flipped image)
#
# Copyright (c) 2006-2012 Sanghyuk Lee and David G. Grier
#
# UPDATES:
#    The most recent version of this program may be obtained from
#    http://physics.nyu.edu/grierlab/software.html
# 
# LICENSE:
#    This program is free software; you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as
#    published by the Free Software Foundation# either version 2 of the
#    License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful, 
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
#    02111-1307 USA
#
#    If the Internet and WWW are still functional when you are using
#    this, you should be able to access the GPL here: 
#    http://www.gnu.org/copyleft/gpl.html
#-

####
# Import Libraries
#
import numpy as np
from numpy import flipud, fliplr, exp, imag, real
from time import *
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from gpu_fft_functions import make_gpu_fftshift, fft, make_thisE_fun, make_multiply_fun
from pycuda import cumath
import skcuda.fft as cu_fft

t0 = -1
import time
def tic():
  global t0
  t0 = time.time()
def toc():
  print time.time() - t0


def idl_ifft(dat, axis = None): #citation
    """
    Calculate FFT the same as IDL
    Converting Numpy = [0,99,98,...,1] -----> IDL = [0,1,2,...,99]
    For 2D data, the x and y axes are flipped in a similar way.
    FIXME(JS): Add implementation for 3D data
    """
    
    if dat.ndim == 1:
        dat  = np.fft.ifft(dat)
        temp = dat[1:len(dat)].copy()
        dat[1:len(dat)] = temp[::-1]    # reverse array
    elif dat.ndim == 2:
        shape = dat.shape
        if axis == 0:
            dat  = np.fft.ifft(dat, axis=axis)
            temp = flipud(dat[1:shape[0],:]).copy() # flip x-axis
            dat[1:shape[0],:] = temp
        elif axis == 0:
            dat  = np.fft.ifft(dat, axis=axis)
            temp = fliplr(dat[:,1:shape[1]]).copy() # flip y-axis
            dat[:,1:shape[1]] = temp
        else:
            dat  = np.fft.ifftn(dat, axes=axis)
            temp = flipud(dat[1:shape[0],:]).copy() # flip x-axis
            dat[1:shape[0],:] = temp
            temp = fliplr(dat[:,1:shape[1]]).copy() # flip y-axis
            dat[:,1:shape[1]] = temp
    
    return dat

def idl_fft(dat, axis = None):  #citation
    """
    Calculate FFT the same as IDL
    Converting Numpy = [0,99,98,...,1] -----> IDL = [0,1,2,...,99]
    For 2D data, the x and y axes are flipped in a similar way.
    FIXME(JS): Add implementation for 3D data
    """
    
    if dat.ndim == 1:
        dat = np.fft.fft(dat)
        temp = dat[1:len(dat)].copy()
        dat[1:len(dat)] = temp[::-1]    # reverse array
    elif dat.ndim == 2:
        shape = dat.shape
        if axis == 0:
            dat = np.fft.fft(dat, axis=axis)
            temp = flipud(dat[1:shape[0],:]).copy() # flip x-axis
            dat[1:shape[0],:] = temp
        elif axis == 0:
            dat = np.fft.fft(dat, axis=axis)
            temp = fliplr(dat[:,1:shape[1]]).copy() # flip y-axis
            dat[:,1:shape[1]] = temp
        else:
            dat = np.fft.fftn(dat, axes=axis)
            temp = flipud(dat[1:shape[0],:]).copy() # flip x-axis
            dat[1:shape[0],:] = temp
            temp = fliplr(dat[:,1:shape[1]]).copy() # flip y-axis
            dat[:,1:shape[1]] = temp
    
    return dat

def hanning(nx, ny):
  """
  Calculates the Hanning Window of size (nx,ny)
  """ 
  if ny <= 0:
    print "Array dimensions must be >= 0"
    return
  if nx <= 0:
    print "Array dimensions must be >= 0"
    return
  row_window = .5*(1-cos(2*pi*arange(0,int(nx))/nx))
  col_window = .5*(1-cos(2*pi*arange(0,int(ny))/ny))
  if ny > 0:
    return outer(row_window,col_window)
  else:
    return row_window  

def rayleighsommerfeld(a, z, lamb = 0.632, mpp = 0.135, nozphase = '', hanning_win = ''): 
  """
  Computes Rayleigh-Sommerfeld back-propagation of a normalized 
  hologram of the type measured by digital video microscopy.

  INPUTS:
  a: hologram recorded as image data normalized
     by a background image. 
  z: displacement from the focal plane [pixels]
     If z is an array of displacements, the field
     is computed at each plane.

  KEYWORDS:
     lamb: Wavelength of light in medium [micrometers]
         Default: 0.632 -- HeNe in air
     mpp: Micrometers per pixel
         Default: 0.135
  
  Example: 
  >>> E   = rayleighsommerfeld(image,z)
  >>> int = abs(E)**2
  """

  a = np.array(a)
  if type(z) == int:
    z = [z]
  z = np.array(z)
  
  # hologram dimensions
  ndim = a.ndim
  if ndim > 2 : 
    print "requires two-dimensional hologram"
  nx = float(a.shape[1])
  if ndim == 1 : 
    ny = 1.
  else:
    ny = float(a.shape[0])
  
  # volumetric slices
  
  if type(z) == int:         # number of z planes
    nz = 1
  else:
    nz = len(z)                
  ci = complex(0., 1.)
  k = 2.*np.pi*mpp/lamb           # wavenumber in radians/pixels
  #ikz = ci*k*z

  # phase factor for Rayleigh-Sommerfeld propagator in Fourier space
  # Refs. [2] and [3]
  qx = np.arange(nx)/nx - 0.5
  qx = ((lamb/mpp) * qx)**2

  if ndim == 2 :
    qy = np.arange(ny)/ny - 0.5
    qy = ((lamb/mpp)*qy)**2
    qsq = np.zeros([ny,nx],dtype = complex)
    for i in range(0,int(nx)):
      qsq[:,i] += qx[i]
    for j in range(0,int(ny)):
      qsq[j,:] += qy[j]
  else:
    qsq = qx

  qfactor = k * np.sqrt(1. - qsq)

  if nozphase != 'nozphase':
    qfactor -= k

  if hanning_win  != '':
    qfactor *= Hanning(ny, nx)

  ikappa = ci * real(qfactor)
  gamma = imag(qfactor) 
  
  a = np.array(a,dtype=complex)

  E = idl_ifft(a-1.) # Fourier transform of input field
  E = np.fft.fftshift(E)

  # Transfer to gpu
  E = np.asarray(E, np.complex64, order='C')
  E_gpu = gpuarray.to_gpu(E)
  ikappa = np.asarray(ikappa, np.complex64, order='C')
  ikappa_gpu = gpuarray.to_gpu(ikappa)
  gamma = np.asarray(gamma, np.complex64, order='C')
  gamma_gpu = gpuarray.to_gpu(gamma)
  gamma = np.asarray(gamma, np.complex64, order='C')
  gamma_gpu = gpuarray.to_gpu(gamma)
  z = np.asarray(z, np.float32, order='C')
  z_gpu = gpuarray.to_gpu(z)

  shape3D = (len(z), E.shape[0], E.shape[1])
  fftshift = make_gpu_fftshift(shape3D)
  thisE_fun = make_thisE_fun(shape3D)
  multiply_fun = make_multiply_fun(shape3D)

  thisE = gpuarray.empty(shape3D, np.complex64)
  thisE_fun(thisE, z_gpu, ikappa_gpu, gamma_gpu, np.uint32(thisE.size))

  Hqz = cumath.exp(thisE)
  multiply_fun(thisE, Hqz, E_gpu, np.uint32(thisE.size))

  fftshift(Hqz, thisE.gpudata, np.uint32(thisE.size))

  plan_forward = cu_fft.Plan(E.shape, np.complex64, np.complex64, len(z))
  cu_fft.fft(Hqz, thisE, plan_forward)

  res = thisE.get()

  return np.transpose(res, (1,2,0))
