# Reading JMA radar data in netcdf format
# take statistics for the whole Japanese region
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
def ext_nc_JMA_all(fname):
    try:
        nc = netCDF4.Dataset(fname, 'r')
    except:
        print("netCDF read error")
        return None
    eps = 0.001 # small number
    #
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

    Rclip_list = []
    for n in range(slct_id.shape[0]):
        ii,jj = slct_id[n,:]
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
        lon_clip=lons.data[i0:i1]
        lat_clip=lats.data[j0:j1]
        Rclip_list.append(Rclip)
    # stack in 0 dimension
    Rclip_grd = np.stack(Rclip_list,axis=0)
    return(Rclip_grd)

def fname_1h_ago(fname):
    f2 = fname.split("/")[-1]
    f2 = f2.replace("2p-jmaradar5_","").replace("utc.nc.gz","")
    dt = datetime.datetime.strptime(f2,'%Y-%m-%d_%H%M')
    
    # +1h data for Y
    date1 = dt - pd.offsets.Hour()
    #fname1 = date1.strftime('../data/jma_radar/%Y/%m/%d/2p-jmaradar5_%Y-%m-%d_%H%Mutc.nc.gz')
    fname1 = date1.strftime('/data/nas_data/jma_radar/%Y/2p-jmaradar5_%Y-%m-%d_%H%Mutc.nc.gz')
    return fname1

def prep_1h_grid(year,infile_root,outfile_root):
    print("preparing for year= ",year)

    # prep dirs
    for n in range(slct_id.shape[0]):
        ii,jj = slct_id[n,:]
        ijstr = "IJ_%d_%d" % (ii,jj)
        ijdir = outfile_root+ijstr+"/"
        # create result dir
        if not os.path.exists(outfile_root+ijstr):
            os.mkdirs(outfile_root+ijstr, exist_ok=True)
    
    for infile in sorted(glob.iglob(infile_root + '*/*/*00utc.nc.gz')):
        # read 1hour data at a time
        # initialize with -999.0
        R_list = []
        for i in range(12):
            shour = '{0:02d}utc'.format(i*5)
            in_zfile = infile.replace('00utc',shour)
            print('reading zipped file:',in_zfile)
            # '-k' option for avoiding removing gz file
            subprocess.run('gunzip -kf '+in_zfile,shell=True)
            in_nc=in_zfile.replace('.gz','')
            print('reading nc file:',in_nc)
            if os.path.exists(in_nc):
                R_list.append(ext_nc_JMA_all(in_nc))
                if R_list is None:
                    continue
            else:
                print('nc file not found!!!',in_nc)
                continue
            subprocess.run('rm '+in_nc,shell=True)
            
        # create 1h data
        print("number of data in an hour",len(R_list))
        if len(R_list) != 12:
            print("data is short; skipped")
            continue
            
        Rhour = np.stack(R_list,axis=0)
        R1h = Rhour
        R1h = R1h.astype("float16")

        for n in range(slct_id.shape[0]):
            # save 1h data as h5 file
            ii,jj = slct_id[n,:]
            ijstr = "IJ_%d_%d" % (ii,jj)
            h5fname = infile.split("/")[-1]
            h5fname = "fulldata_"+h5fname.replace('.nc.gz','_'+str(ii)+'_'+str(jj)+'.h5')
            print('writing h5 file:',h5fname,np.max(R1h[:,n,:,:]))
            ijdir = outfile_root+ijstr+"/"
            h5file = h5py.File(ijdir+h5fname,'w')
            h5file.create_dataset("R",data= R1h[:,n,:,:])
            h5file.close() 

if __name__ == '__main__':
    # read case name from command line
    argvs = sys.argv
    argc = len(argvs)
    
    if argc != 2:
        print('Usage: python prep_radarJMA_fulldata_alljapan.py YYYY ')
        quit()
    year = int(argvs[1])
    
    #for year in [2015,2016,2017]:
    #year = 2015
    #year = 2016
    #year = 2017
    #year = 2010
    #year = 2011
    #year = 2012
    #year = 2013
    #year = 2014
    #year = 2018
    #year = 2019


    # read
    infile_root = "/data/nas_data/jma_radar/%d/" % (year)
    print('input dir:',infile_root)

    # outfile
    outfile_root = '../data/data_alljapan_fulldata/'
    print('output dir:',outfile_root)

    prep_1h_grid(year,infile_root,outfile_root)


