#---------------------------------------------------
# Generate Topography Data
#---------------------------------------------------
import glob
import subprocess
import sys
import os.path
from osgeo import gdal

import netCDF4
import numpy as np
import pandas as pd
import h5py
import elevation

import shapefile

import matplotlib
import matplotlib.pyplot as plt

import scipy.ndimage
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator

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

def grid_ij_jma_radar(ii,jj):
    '''
    This routine returns lon-lat grid, which is consistent with JMA radar data
    around IJ specified Region in Japan
    '''
    #nc = netCDF4.Dataset('/data/nas_data/jma_radar/2015/01/01/2p-jmaradar5_2015-01-01_0000utc.nc', 'r')
    nc = netCDF4.Dataset('../data/temp/2p-jmaradar5_2015-01-01_0000utc.nc', 'r')
    #
    # dimensions
    nx = len(nc.dimensions['LON'])
    ny = len(nc.dimensions['LAT'])
    nt = len(nc.dimensions['TIME'])
    print("JMA Radar data dims:",nx,ny,nt)
    #
    # ext variable
    lons = nc.variables['LON'][:]
    lats = nc.variables['LAT'][:]
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
    #
    # offset
    ii = ii - 8
    jj = jj - 8
    #
    ii0 = i0 + nx_clip*ii
    jj0 = j0 + ny_clip*jj
    ii1 = ii0 + nx_clip
    jj1 = jj0 + ny_clip
    # extract data
    lon_clip=lons.data[ii0:ii1]
    lat_clip=lats.data[jj0:jj1]
    return lon_clip, lat_clip

def read_hrncst_smoothed(lons,lats,R,lons_ij,lats_ij):
    '''
    Read high resolution nowcast data and output in a smoothed grid

    '''
    # take min-max range with some margin
    delta = 1.0
    lon_min = np.min(lons_ij) - delta
    lon_max = np.max(lons_ij) + delta
    lat_min = np.min(lats_ij) - delta
    lat_max = np.max(lats_ij) + delta

    id_lons = (lons < lon_max) * (lons > lon_min)
    id_lats = (lats < lat_max) * (lats > lat_min)
    lons_rect = lons[id_lons]
    lats_rect = lats[id_lats]
    # "R[:,id_lats,id_lons]" does not seem to work..
    r_tmp =R[:,id_lats,:]
    r_rect =np.array(r_tmp[:,:,id_lons])
    r_rect = np.maximum(r_rect,0) # replace negative value with 0
    del r_tmp

    # if outside region, return nan
    if (np.min(lons_ij) < np.min(lons)) or (np.max(lons_ij) > np.max(lons)):
        print("outside region: skipped") 
        return None
    if (np.min(lats_ij) < np.min(lats)) or (np.max(lats_ij) > np.max(lats)):
        print("outside region: skipped") 
        return None
    
    tdim = r_rect.shape[0] # time dimension for nowcast
    r_out = np.zeros((tdim,len(lats_ij),len(lons_ij)))
    for i in range(tdim):
        # Apply gaussian filter (Smoothing)
        sigma = [0.01, 0.01] # smooth 250m scale to 1km scale
        r_sm = scipy.ndimage.filters.gaussian_filter(r_rect[i,:,:], sigma, mode='constant')
        #r_sm = r_rect[i,:,:]
        # Interpolate by nearest neighbour
        intfunc = RegularGridInterpolator((lats_rect, lons_rect), r_sm)
        la2, lo2 = np.meshgrid(lats_ij, lons_ij)
        pts = np.vstack([la2.flatten(),lo2.flatten()])
        r_interp = intfunc(pts.T)
        r_interp = r_interp.reshape([len(lats_ij),len(lons_ij)]).T
        r_out[i,:,:] = r_interp
    return r_out

if __name__ == '__main__':
    
    #download_flg = True
    download_flg = False

    # read JMA Climate Data
    fcsv = "../data/JMA_climate/climate_JMA_Rainfall_alllist.csv"
    if download_flg:
        csvfile = open(fcsv, "w")
        print("id,lon,lat,rainfall",file=csvfile)
    
        paths = glob.glob("../data/JMA_climate/*GML/*shp")
        for path in paths:
            print("path:",path)
        
            src = shapefile.Reader(path,encoding='SHIFT-JIS')

            print ("No. of records:",len(src))
            for n in range(len(src)):
                shp = src.shape(n)
                recs = src.record(n)
                # extract lon lat
                lon = shp.points[0][0]
                lat = shp.points[0][1]
                # extract value
                id = recs[0]
                rainfall = recs[13]/10.0 # to [mm]
                print ("record :",id,lon,lat,rainfall)
                print("%s,%f,%f,%f" % (id,lon,lat,rainfall),file=csvfile)
            
    # Obtain mean precipitation for each region

    fcsv_out = "../data/JMA_climate/stat_climate_JMA.csv"
    csvout = open(fcsv_out, "w")
    print("ii,jj,ijstr,rain_mean,rain_std,samples",file=csvout)
    
    df = pd.read_csv("../data/JMA_climate/climate_JMA_Rainfall_alllist.csv")

    for n in range(slct_id.shape[0]):
        ii,jj = slct_id[n,:]
        ijstr = "IJ_%d_%d" % (ii,jj)
        lons_ij, lats_ij = grid_ij_jma_radar(ii,jj)

        # take rectangular area
        dll = 0.3
        lon_min = np.min(lons_ij)-dll
        lon_max = np.max(lons_ij)+dll
        lat_min = np.min(lats_ij)-dll
        lat_max = np.max(lats_ij)+dll

        df_slct = df.loc[(df['lon'] > lon_min) & (df['lon'] < lon_max) &
                         (df['lat'] > lat_min) & (df['lat'] < lat_max)]
        
        if(len(df_slct)>0):
            N = len(df_slct)
            print("N=",N)
            rain_mean = df_slct['rainfall'].mean()
            rain_std = df_slct['rainfall'].std()
        else:
            N = 0
            print("N=",N)
            rain_mean = 0.0
            rain_std = 0.0
            
        # write to csv file
        print("%d,%d,%s,%f,%f,%d" % (ii,jj,ijstr,rain_mean,rain_std,N),
              file=csvout)
    
