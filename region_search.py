# -*- coding: utf-8 -*-
"""
region_search.py
------------------------------------------------------------------
Finds the largest connected area in grid
"""
import numpy as np
import os

import numpy as np
from scipy.ndimage import label

#np.set_printoptions(threshold=np.NaN)


def find_biggest_region_old(data):
	region_inf=np.zeros(shape=(31,31,31))
	region_flag=[0]
	for ii in range(1,30):
		for jj in range(1,30):
			for zz in range(1,30):
				if data[ii,jj,zz]==1:
					neighbor=np.array([region_inf[ii-1,ll,mm] for ll in range(jj-1,jj+2) for mm in range(zz-1,zz+2)]+[region_inf[ii,jj-1,zz-1],region_inf[ii,jj,zz-1],region_inf[ii,jj-1,zz]])
					valid_neighbor=neighbor[neighbor!=0]
					if len(valid_neighbor)==0:
						region_inf[ii,jj,zz]=max(region_flag)+1
						region_flag.append(max(region_flag)+1)
					if len(valid_neighbor)!=0:
						new_flag=min(valid_neighbor)
						region_inf[ii,jj,zz]=new_flag
						for flag in set(valid_neighbor):
							region_inf[region_inf==flag]=new_flag
							region_flag.remove(flag)
						region_flag.append(new_flag)

	num=0
	region_flag.remove(0)
	
	if len(region_flag)==0:
		return data,0

	biggest_region=region_flag[0]
	for flag in region_flag:
		temp_num=np.sum(region_inf==flag)
		if temp_num>num:
			num=temp_num
			biggest_region=flag

	out=np.zeros(shape=(31,31,31),dtype=int)
	out[region_inf==biggest_region]=1
	return out,len(region_flag)


def find_largest_connected_region(voxel: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Find the largest connected region of 1's in a 3D grid using 26-connectivity.

    Parameters
    ----------
    voxel : ndarray
        Binary 3D voxel grid (1 = filled voxel, 0 = empty).

    Returns
    -------
    largest_region : ndarray
        Binary grid containing only the largest connected component.
    num_regions : int
        Total number of connected regions found.
    """
    # Define 26-connectivity: all neighbors in 3x3x3 cube excluding center
    structure = np.ones((3, 3, 3), dtype=int)

    labeled, num_features = label(voxel, structure=structure)

    if num_features == 0:
        return np.zeros_like(voxel, dtype=int), 0

    # Compute the size of each region
    sizes = np.bincount(labeled.ravel())

    # Ignore background (label 0)
    sizes[0] = 0
    largest_label = sizes.argmax()

    # Return only the largest connected region
    largest_region = (labeled == largest_label).astype(int)

    return largest_region #,num_features





