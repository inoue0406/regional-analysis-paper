#---------------------------------------------------
# Preprocess high-resolution nowcast data by JMA
#---------------------------------------------------
import glob
import subprocess
import sys
import os.path

import netCDF4
import numpy as np
import h5py

import scipy.ndimage
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator

# pre-selected 
slct_id = np.array([#[2, 1],
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

    # read case name from command line
    argvs = sys.argv
    argc = len(argvs)

    if argc != 2:
        print('Usage: python plot_comp_prediction.py YYYY-DD')
        quit()
    year_day = argvs[1]

    # read
    #infile_root = '/data/nas_data/jma_nowcast/4p-hrncstprate/'
    infile_root = '/inoue_home_nas/data/jma_nowcast/4p-hrncstprate/'
    
    #infile_root = '../data/4p-hrncstprate_rerun/'
    print('input dir:',infile_root)

    nx = 200
    ny = 200
    nt = 7
    
    # process only 00min file
    #file_list = sorted(glob.iglob(infile_root + '/*00utc.nc.gz'))
    # process all the file
    #file_list = sorted(glob.iglob(infile_root + '/*utc.nc.gz'))
    file_list = sorted(glob.iglob(infile_root + '/4p-hrncstprate_japan0250_'+year_day+'*utc.nc.gz'))

    # initializa lons_ij, lats_ij pair
    lonlat_list = []
    for n in range(slct_id.shape[0]):
        ii,jj = slct_id[n,:]
        ijstr = "IJ_%d_%d" % (ii,jj)
        lons_ij, lats_ij = grid_ij_jma_radar(ii,jj)
        lonlat_list.append([lons_ij,lats_ij])

    # restart
    # file_list = file_list[4350:]
    for infile in file_list:
        # read 1hour data at a time
        # initialize with -999.0
        R1h = np.full((nt,nx,ny),-999.0,dtype=np.float32)

        in_zfile = infile
        in_zfile_cp = in_zfile.replace(infile_root,'../data/temp/')
        subprocess.run('cp '+in_zfile+' '+in_zfile_cp,shell=True)
    
        print('reading zipped file:',in_zfile_cp)
        # '-k' option for avoiding removing gz file
        subprocess.run('gunzip -kf '+in_zfile_cp,shell=True)
        in_nc=in_zfile_cp.replace('.gz','')
        
        print('reading nc file:',in_nc)
        if os.path.exists(in_nc):

            # read the whole area
            nc = netCDF4.Dataset(in_nc, 'r')
            # dimensions
            nx = len(nc.dimensions['LON'])
            ny = len(nc.dimensions['LAT'])
            nt = len(nc.dimensions['TIME'])
            print("dims:",nx,ny,nt)
            # extract variable
            lons = np.array(nc.variables['LON'][:])
            lats = np.array(nc.variables['LAT'][:])
            R = np.array(nc.variables['PRATE'][:]) # numpy.ma.core.MaskedArray
            # long_name: precipitation rate
            # units: 1e-3 meter/hour -> [mm/h]
            # scale_factor: 0.01
            # add_offset: 0.0

            for n in range(slct_id.shape[0]):
                ii,jj = slct_id[n,:]
                ijstr = "IJ_%d_%d" % (ii,jj)
                # outfile
                outfile_root = '../data/hrncst_fulldata/'+ijstr+"/"
                print('output dir:',outfile_root)

                if not os.path.exists(outfile_root):
                    os.mkdir(outfile_root)

                #lons_ij, lats_ij = grid_ij_jma_radar(ii,jj)
                lons_ij = lonlat_list[n][0]
                lats_ij = lonlat_list[n][1]
            
                R1h = read_hrncst_smoothed(lons,lats,R,lons_ij,lats_ij)
                if R1h is None:
                    # remove stray file
                    subprocess.run('rm ../data/temp/4p-hrncst*',shell=True)
                    continue
                # write to h5 file
                h5fname = infile.split('/')[-1]
                h5fname = h5fname.replace('.nc.gz','.h5')
                print('writing h5 file:',h5fname)
                h5file = h5py.File(outfile_root+h5fname,'w')
                h5file.create_dataset('R',data= R1h.astype("float16"),
                                      compression="gzip")
            del lons,lats,R,nc
        else:
            print('nc file not found!!!',in_nc)
            # remove stray file
            subprocess.run('rm ../data/temp/4p-hrncst*',shell=True)
            continue
        if os.path.exists(in_zfile_cp):
            os.remove(in_zfile_cp)
        if os.path.exists(in_nc):
            os.remove(in_nc)
        #subprocess.run('rm '+in_zfile_cp,shell=True)
        #subprocess.run('rm '+in_nc,shell=True)

