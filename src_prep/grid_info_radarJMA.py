# extract grid information
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py

import glob
import gzip
import subprocess
import sys
import os.path

import datetime

# pre-selected 
slct_id = np.array([[2, 1],
                    [2, 2],
                    [3, 2],
                    [3, 3],
                    [4, 2],
                    [4, 3],
                    [4, 4],
                    [4, 5],
                    [4, 6],
                    [4, 7],
                    [5, 6],
                    [5, 7],
                    [5, 8],
                    [5, 9],
                    [6, 6],
                    [6, 7],
                    [6, 8],
                    [6, 9],
                    [7, 7],
                    [7, 8],
                    [7, 9],
                    [7, 10],
                    [8, 8],
                    [8, 9],
                    [8, 10],
                    [8, 11],
                    [8, 12],
                    [8, 13],
                    [9, 9 ],
                    [9, 10],
                    [9, 11],
                    [9, 12],
                    [9, 13],
                    [10, 12],
                    [10, 13]])

# extract data from nc file
def ext_grid_info_all(fname):
    try:
        nc = netCDF4.Dataset(fname, 'r')
    except:
        print("netCDF read error")
        return None
    # dimensions
    nx = len(nc.dimensions['LON'])
    ny = len(nc.dimensions['LAT'])
    nt = len(nc.dimensions['TIME'])
    print("dims:",nx,ny,nt)
    #
    # ext variable
    lons = nc.variables['LON'][:]
    lats = nc.variables['LAT'][:]
    R = nc.variables['PRATE'][:]
    #
    # clip area around tokyo
    lat_tokyo = 35.681167
    lon_tokyo = 139.767052
    nx_clip = 200
    ny_clip = 200
    i0=np.argmin(np.abs(lons.data-lon_tokyo)) - int(nx_clip/2)
    j0=np.argmin(np.abs(lats.data-lat_tokyo)) - int(ny_clip/2)
    i1=i0+nx_clip
    j1=j0+ny_clip
    # check
    if((i0 != 1641) or(j0 != 1781)):
        print("irregular i0,j0: skip.. ",i0,j0)
        return None

    area_list = []
    for n in range(slct_id.shape[0]):
        ii,jj = slct_id[n,:]
        ijstr = "IJ_%d_%d" % (ii,jj)
        # choose ii and jj to span the whole dataset area
        Rslct = np.zeros(35)
        count = 0
        # offset
        ii = ii - 8
        jj = jj - 8
        #
        ii0 = i0 + nx_clip*ii
        jj0 = j0 + ny_clip*jj
        ii1 = ii0 + nx_clip
        jj1 = jj0 + ny_clip
        # extract data
        Rclip = R.data[0,jj0:jj1,ii0:ii1]
        Rclip = Rclip.T   # transpose so that i index comes first
        lon_clip=lons.data[ii0:ii1]
        lat_clip=lats.data[jj0:jj1]

        #ngrd = len(lon_clip)
        #df_grid = pd.DataFrame({"ii":[ii+8]*ngrd,
        #                         "jj":[jj+8]*ngrd,
        #                         "ijstr":[ijstr]*ngrd,
        #                         "lon":lon_clip,
        #                         "lat":lat_clip})
        df_area = pd.DataFrame({"ii":[ii+8],
                                 "jj":[jj+8],
                                 "ijstr":[ijstr],
                                 "lon_min":np.min(lon_clip),
                                 "lon_max":np.max(lon_clip),
                                 "lat_min":np.min(lat_clip),
                                 "lat_max":np.max(lat_clip)})
        area_list.append(df_area)
    # stack in 0 dimension
    df_area_all = pd.concat(area_list)
    df_area_all.to_csv("../data/jmaradar_grid_lonlat_list.csv",index=False)

if __name__ == '__main__':

    year = 2015
    infile_root = "/data/nas_data/jma_radar/%d/" % (year)
    print('input dir:',infile_root)
    
    for infile in sorted(glob.iglob(infile_root + '*/*/*00utc.nc.gz')):
        print('reading zipped file:',infile)
        # '-k' option for avoiding removing gz file
        subprocess.run('gunzip -kf '+infile,shell=True)
        in_nc=infile.replace('.gz','')
        print('reading nc file:',in_nc)
        ext_grid_info_all(in_nc)
