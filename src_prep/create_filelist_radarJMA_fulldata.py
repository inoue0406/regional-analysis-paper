#!/usr/bin/env python

# parse data directory to make train/test list

import glob
import pandas as pd
import numpy as np
import os

import pdb
import h5py

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

def create_filelist(dirpath,ijstr,dir_out,setting):
    # initialize
    max_list = []
    min_list = []
    mean_list = []
    fnames = []
    fnexts = []
    
    if setting == "train":
        # select 2015 and 2016 for training
        dates = pd.date_range(start='2015-01-01 00:00', end='2016-12-31 22:00', freq='H')
    elif setting == "valid":
        # select 2017 for validation
        dates = pd.date_range(start='2017-01-01 00:00', end='2017-12-31 22:00', freq='H')
        
    # file format "2p-jmaradar5_2015-01-01_0000utc.h5"
    
    # We choose loop through continuous times for missed-file checking and 
    # checking for valid X-Y pairs
    
    for n,date in enumerate(dates):
        ij = ijstr.replace("IJ_","")
        print(date)
        fname = date.strftime('fulldata_2p-jmaradar5_%Y-%m-%d_%H%Mutc_'+ij+'.h5')
        #print(fname)
        fpath = os.path.join(dirpath,fname)
        
        # +1h data for Y
        date1 = date + pd.offsets.Hour()
        fname1 = date1.strftime('fulldata_2p-jmaradar5_%Y-%m-%d_%H%Mutc_'+ij+'.h5')
        fpath1 = os.path.join(dirpath,fname1)
    
        # current data
        if os.path.exists(fpath) and os.path.exists(fpath1):
            #print('Exists:',fpath,fpath1)
            fnames.append(fname)
            fnexts.append(fname1)
        else:
            print('Not exist !!:',fpath,fpath1)
    
    df = pd.DataFrame({'fname':fnames,'fnext':fnexts})
    
    if setting == "train":
        df.to_csv(dir_out + '/train_JMARadar_'+ijstr+'.csv')
    elif setting == "valid":
        df.to_csv(dir_out + '/valid_JMARadar_'+ijstr+'.csv')

if __name__ == '__main__':

    dir_out = "../data/filelist_fulldata"
    setting = "train"
    #setting = "valid"
    
    for n in range(slct_id.shape[0]):
        ii,jj = slct_id[n,:]
        ijstr = "IJ_%d_%d" % (ii,jj)
        dirpath = "../data/data_alljapan_fulldata/"+ijstr+"/"
        create_filelist(dirpath,ijstr,dir_out,setting)
    #pdb.set_trace()
        
