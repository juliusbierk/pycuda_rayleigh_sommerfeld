import numpy as np
import matplotlib.pyplot as plt
from gpu_rs import rayleighsommerfeld
from skimage import io
import matplotlib.cm as cm
import glob
import cPickle
from scipy.ndimage import convolve1d
import glob
import gc
from multiprocessing import Pool
import os 

t0 = -1
import time
def tic():
  global t0
  t0 = time.time()
def toc():
  print time.time() - t0

######### parameters
speed_up = 1
z = np.linspace(0,300/speed_up,80)
#############

fname = 'testimg.png'
# Read image
a = io.imread(fname)

# Do Rayleigh Sommerfeld
a = a[::speed_up,::speed_up] # for speed while developing
print 'Running', fname
tic()
out = np.real(rayleighsommerfeld(a, z, lamb=0.5))
print 'Full:'
toc()

if True:
	os.mkdir('recon')
	for zi, zp in enumerate(z):
		print 'Printing image', zi
		plt.clf()
		plt.subplot(1,2,1)
		plt.imshow(a.T, cmap=cm.gray, vmin=50, vmax=200)

		plt.subplot(1,2,2)
		img = np.flipud(np.abs(out[:,:,zi].T)**2)
		plt.imshow(img, cmap=cm.gray, origin='lower', vmin=50**2, vmax=200**2)
		plt.title('reconstruction depth z = '+"%4.2f" % zp)
		plt.savefig('recon/%i.png'%zi)
