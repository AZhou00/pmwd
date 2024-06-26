import numpy as np

def filename(index):
    # format string by changing the run number to index in 3 digit integers
    # '/ocean/projects/phy230060p/junzhez/datasets/kappaTNG_small/kappaTNG_samples/kappaTNG-Dark/run001/kappa23.dat'
    if index < 1 or index > 100:
        raise ValueError('index should be between 1 and 100 (inclusive)')
    return '/hildafs/projects/phy230056p/junzhez/data/kappaTNG_small/kappaTNG_samples/kappaTNG-Dark/run{}/kappa23.dat'.format(str(index).zfill(3))

def load(fname):
    ng = 1024  # number of grids
    theta = 5.0  # opening angle in deg

    pix_size = theta/ng  # pixel size
    theta = pix_size*np.arange(ng)


    with open(fname, 'rb') as f:
        dummy = np.fromfile(f, dtype="int32", count=1)
        kappa = np.fromfile(f, dtype="float", count=ng*ng)
        dummy = np.fromfile(f, dtype="int32", count=1)

    kappa = kappa.reshape((ng, ng))

    return kappa, pix_size # in degree