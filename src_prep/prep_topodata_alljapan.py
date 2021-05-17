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
import h5py
import elevation

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

    nx = 200
    ny = 200
    
    topo_dir = "../data/topo_SRTM"
    #download_flg = True
    download_flg = False

    # initializa lons_ij, lats_ij pair
    lonlat_list = []
    for n in range(slct_id.shape[0]):
        ii,jj = slct_id[n,:]
        ijstr = "IJ_%d_%d" % (ii,jj)
        lons_ij, lats_ij = grid_ij_jma_radar(ii,jj)
        lonlat_list.append([lons_ij,lats_ij])

        # Download STRM Data
        if download_flg:
            dt = 0.2
            x1 = np.min(lons_ij)+dt
            y1 = np.min(lats_ij)+dt
            x2 = np.max(lons_ij)-dt
            y2 = np.max(lats_ij)-dt
            command = "eio clip -o %s/Shasta-30m-DEM_%s.tif --bounds %f %f %f %f" % \
                      (topo_dir,ijstr,x1,y1,x2,y2)
            print(command)
            subprocess.run(command,shell=True)
        
    # read topo data

    fcsv = "%s/stat_topo_SRTM.csv" % (topo_dir)
    csvfile = open(fcsv, "w")
    print("ii,jj,ijstr,elevation_mean,elevation_max,slope",file=csvfile)
    
    for n in range(slct_id.shape[0]):
        ii,jj = slct_id[n,:]
        ijstr = "IJ_%d_%d" % (ii,jj)
        lons_ij, lats_ij = grid_ij_jma_radar(ii,jj)
        fpath = "%s/Shasta-30m-DEM_%s.tif" % (topo_dir,ijstr)
        print("reading topo data:",fpath)

        if os.path.exists(fpath):
            gdal_data = gdal.Open(fpath)
            gdal_band = gdal_data.GetRasterBand(1)
            nodataval = gdal_band.GetNoDataValue()
            
            data_array = gdal_data.ReadAsArray().astype(np.float)

            # take statistics for data
            data_array[data_array <0.0]=0.0 # negative -> zero
            elevation_mean = np.mean(data_array)
            elevation_max = np.max(data_array)
            # slopes
            sl_y,sl_x = np.gradient(data_array,0.2,0.2)
            slope = np.sqrt(sl_y**2 + sl_x**2)
            slope_mean = np.mean(slope)
            #import pdb;pdb.set_trace()
            print("%d,%d,%s,%f,%f,%f" % (ii,jj,ijstr,elevation_mean,elevation_max,slope_mean),
                  file=csvfile)
            
            #Plot our data with Matplotlib's 'contourf'
            fig, ax = plt.subplots(figsize=(8,8))
            plt.contourf(data_array[::-1,:], cmap = "terrain",
                         levels = list(np.linspace(0.01, 1000, 50)))#,origin="lower")
            plt.title("Elevation Contour:%s" % ijstr)
            cbar = plt.colorbar()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()
            fig.savefig("%s/Shasta-30m-DEM_%s_tstout.png" % (topo_dir,ijstr))
            plt.close()

        #import pdb;pdb.set_trace()

    #elevation.clip(bounds=(123.0, 23.0, 124.0, 24.0), output=path)
        


