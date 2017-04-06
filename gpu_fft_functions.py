import numpy as np 
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from skcuda import fft as cu_fft
from pycuda.compiler import SourceModule
from string import Template
import skcuda.misc as misc
from functools import partial

def make_gpu_fftshift(shape):
	dimZ, imx, imy = shape
	
	mod = Template("""
	#define INDEX(a,b,c) a*${B}*${C}+b*${C}+c
	#define SHIFTINDEX(a,b,c) a*${B}*${C}+((b+NB2)%${B})*${C}+((c+NC2)%${C})
	__global__ void func(float2 *res, float2 *x, unsigned int N) {
	    // Obtain the linear index corresponding to the current thread:
	    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
	                       blockIdx.x*${max_threads_per_block}+threadIdx.x;
	    // Convert the linear index to subscripts:
	    unsigned int a = idx/(${B}*${C});
	    unsigned int b = (idx%(${B}*${C}))/${C};
	    unsigned int c = (idx%(${B}*${C}))%${C};

	    int NB2 = ${B}/2;
	    int NC2 = ${C}/2;

	    if (idx < N) {
		    res[INDEX(a,b,c)].x = x[SHIFTINDEX(a,b,c)].x;
		    res[INDEX(a,b,c)].y = x[SHIFTINDEX(a,b,c)].y;
	    }
	}
	""")
	max_threads_per_block, max_block_dim, max_grid_dim = misc.get_dev_attrs(pycuda.autoinit.device)
	block_dim, grid_dim = misc.select_block_grid_sizes(pycuda.autoinit.device, shape)
	max_blocks_per_grid = max(max_grid_dim)

	func_mod = \
	    SourceModule(mod.substitute(max_threads_per_block=max_threads_per_block,
	                                              max_blocks_per_grid=max_blocks_per_grid,
												  A=dimZ, B=imx, C=imy))

	func = func_mod.get_function('func')
	return partial(func, block=block_dim, grid=grid_dim)


def make_thisE_fun(shape):
	dimZ, imx, imy = shape

	mod = Template("""
	#define INDEX3D(a,b,c) a*${B}*${C}+b*${C}+c
	#define INDEX2D(b,c) b*${C}+c 
	__global__ void thisE_fun(float2 *res, float *z, float2 *ikappa, float2 *gamma, unsigned int N) {
	    // Obtain the linear index corresponding to the current thread:
	    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
	                       blockIdx.x*${max_threads_per_block}+threadIdx.x;
	    // Convert the linear index to subscripts:
	    unsigned int a = idx/(${B}*${C});
	    unsigned int b = (idx%(${B}*${C}))/${C};
	    unsigned int c = (idx%(${B}*${C}))%${C};
	    unsigned int i;
	    float2 m1;
	    // Use the subscripts to access the array:
	    if (idx < N) {
        	i = INDEX2D(b,c);
        	m1.x = ikappa[i].x * z[a] - gamma[i].x * abs(z[a]);
        	m1.y = ikappa[i].y * z[a] - gamma[i].y * abs(z[a]);
            res[INDEX3D(a,b,c)] = m1;
	    }
	}
	""")

	max_threads_per_block, max_block_dim, max_grid_dim = misc.get_dev_attrs(pycuda.autoinit.device)
	block_dim, grid_dim = misc.select_block_grid_sizes(pycuda.autoinit.device, shape)
	max_blocks_per_grid = max(max_grid_dim)

	func_mod = \
	    SourceModule(mod.substitute(max_threads_per_block=max_threads_per_block,
	                                              max_blocks_per_grid=max_blocks_per_grid,
												  A=dimZ, B=imx, C=imy))

	func = func_mod.get_function('thisE_fun')
	return partial(func, block=block_dim, grid=grid_dim)

def make_multiply_fun(shape):
	dimZ, imx, imy = shape

	mod = Template("""
	#define INDEX3D(a,b,c) a*${B}*${C}+b*${C}+c
	#define INDEX2D(b,c) b*${C}+c 
	__global__ void multiply_fun(float2 *res, float2 *Hqz, float2 *E, unsigned int N) {
	    // Obtain the linear index corresponding to the current thread:
	    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
	                       blockIdx.x*${max_threads_per_block}+threadIdx.x;
	    // Convert the linear index to subscripts:
	    unsigned int a = idx/(${B}*${C});
	    unsigned int b = (idx%(${B}*${C}))/${C};
	    unsigned int c = (idx%(${B}*${C}))%${C};
	    unsigned int i, j;
	    float2 m1;
	    // Use the subscripts to access the array:
	    if (idx < N) {
        	i = INDEX2D(b,c);
        	j = INDEX3D(a,b,c);
        	m1.x = Hqz[j].x*E[i].x - Hqz[j].y*E[i].y; // complex multiplication
        	m1.y = Hqz[j].x*E[i].y + Hqz[j].y*E[i].x;
            res[j] = m1;
	    }
	}
	""")

	max_threads_per_block, max_block_dim, max_grid_dim = misc.get_dev_attrs(pycuda.autoinit.device)
	block_dim, grid_dim = misc.select_block_grid_sizes(pycuda.autoinit.device, shape)
	max_blocks_per_grid = max(max_grid_dim)

	func_mod = \
	    SourceModule(mod.substitute(max_threads_per_block=max_threads_per_block,
	                                              max_blocks_per_grid=max_blocks_per_grid,
												  A=dimZ, B=imx, C=imy))

	func = func_mod.get_function('multiply_fun')
	return partial(func, block=block_dim, grid=grid_dim)


def fft(data_in, data_out):
	plan_forward = cu_fft.Plan((data_in.shape[0], data_in.shape[1]), np.complex64, np.complex64)
	cu_fft.fft(data_in, data_out, plan_forward)

def ifft(data_in, data_out):
	plan_inverse = cu_fft.Plan((data_in.shape[0], data_in.shape[1]), np.complex64, np.complex64)
	cu_fft.ifft(data_in, data_out, plan_inverse)
