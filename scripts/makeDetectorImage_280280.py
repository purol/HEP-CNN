#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
import sys
import uproot
from glob import glob
from math import ceil
import numba
import numpy, numba, awkward, awkward.numba

if sys.version_info[0] < 3: sys.exit() # major version

parser = argparse.ArgumentParser()
parser.add_argument('input', nargs='+', action='store', type=str, help='input file names')
parser.add_argument('-o', '--output', action='store', type=str, help='output file name', required=True)
parser.add_argument('-n', '--nevent', action='store', type=int, default=-1, help='number of events to preprocess')
parser.add_argument('--width', action='store', type=float, default=280, help='image width, along eta')
parser.add_argument('--height', action='store', type=float, default=280, help='image height, along phi')
parser.add_argument('--format', action='store', choices=('NHWC', 'NCHW'), default='NCHW', help='image format for output (NHWC for TF default, NCHW for pytorch default)')
parser.add_argument('-c', '--chunk', action='store', type=int, default=1024, help='chunk size')
parser.add_argument('--compress', action='store', choices=('gzip', 'lzf', 'none'), default='none', help='compression algorithm')
parser.add_argument('-d', '--debug', action='store_true', default=False, help='debugging')
parser.add_argument('--precision', action='store', type=int, choices=(8,16,32,64), default=32, help='Precision')
parser.add_argument('--mc', action='store_true', default=True, help='flag to set MC sample')
args = parser.parse_args()
# get arguments

###################################################################################################

#@numba.njit(nogil=True, fastmath=True, parallel=True)
## note: numba does not support np.histogram2d. sad...
def getImage(energy, bins): # bins: [x axin bin num, y axis bin num]
    # images_h = getImage(src_towers_Ehad[begin:end], src_towers_eta[begin:end], src_towers_phi[begin:end], [self.width, self.height])
    xlim = [0, 280] # range at x axis
    ylim = [0, 280] # range at y axis

    hs = []

    for i in range(len(energy)):
        x=[]
        y=[]
        value=[]
        for j in range(280):
            for k in range(280):
                x.append(j)
                y.append(k)
                value.append(energy[i][j][k])
        h = np.histogram2d(x, y, weights=value, bins=bins, range=[xlim, ylim])
        hs.append(h[0])

    return np.stack(hs)

###################################################################################################

class FileSplitOut:
    def __init__(self, maxEvent, args):
        self.maxEvent = maxEvent

        self.chunkSize = args.chunk
        self.debug = args.debug

        self.height = args.height
        self.width = args.width
        
        if args.format == 'NHWC': ## TF default
            self.shape = [self.height, self.width, 4]
            self.chAxis = -1
        else: ## pytorch default NCHW
            self.shape = [4, self.height, self.width]
            self.chAxis = 1

        precision = 'f%d' % (args.precision//8)
        self.kwargs = {'dtype':precision}
        if args.compress == 'gzip':
            self.kwargs.update({'compression':'gzip', 'compression_opts':9})
        elif args.compress == 'lzf':
            self.kwargs.update({'compression':'lzf'})

        if not args.output.endswith('.h5'): self.prefix, self.suffix = args.output+'/data', '.h5'
        else: self.prefix, self.suffix = args.output.rsplit('.', 1)

        self.nOutFile = 0
        self.nOutEvent = 0

        self.initOutput()

    def initOutput(self):
        ## Build placeholder for the output
        self.weights = np.ones(0)
        self.images = np.ones([0,*self.shape])

    def addEvents(self, src_weights, Energy_S_first, Energy_C_first, Energy_S_second, Energy_C_second):
        nSrcEvent = len(src_weights) ## the number of events
        begin = 0
        while begin < nSrcEvent:
            end = min(nSrcEvent, begin+self.maxEvent-len(self.weights)) # len(self.weights): the number of data which is saved in buffer currently
            self.nOutEvent += (end-begin)
            print("%d events processed..." % (self.nOutEvent), end='\r')

            self.weights = np.concatenate([self.weights, src_weights[begin:end]])
            # concatenate: link lists

            images_S_first = getImage(Energy_S_first[begin:end], [self.width, self.height])
            images_C_first = getImage(Energy_C_first[begin:end], [self.width, self.height])
            images_S_second = getImage(Energy_S_second[begin:end], [self.width, self.height])
            images_C_second = getImage(Energy_C_second[begin:end], [self.width, self.height])
            
            images_S_first = 1000 * images_S_first
            images_C_first = 1000 * images_C_first
            images_S_second = 1000 * images_S_second
            images_C_second = 1000 * images_C_second

            images_S_first = np.expand_dims(images_S_first, self.chAxis)
            images_C_first = np.expand_dims(images_C_first, self.chAxis)
            images_S_second = np.expand_dims(images_S_second, self.chAxis)
            images_C_second = np.expand_dims(images_C_second, self.chAxis)

            images = np.concatenate([images_S_first, images_C_first, images_S_second, images_C_second], axis=self.chAxis)
            
            self.images  = np.concatenate([self.images, images])

            if len(self.weights) == self.maxEvent: self.flush()
            begin = end

    def flush(self):
        self.save()
        self.initOutput()

    def save(self):
        fName = "%s_%d.h5" % (self.prefix, self.nOutFile)
        nEventToSave = len(self.weights)
        if nEventToSave == 0: return
        if self.debug: print("Writing output file %s..." % fName, end='')

        chunkSize = min(self.chunkSize, nEventToSave)
        with h5py.File(fName, 'w', libver='latest', swmr=True) as outFile:
            g = outFile.create_group('all_events')
            g.create_dataset('weights', data=self.weights, chunks=(chunkSize,), dtype='f4')
            g.create_dataset('images' , data=self.images , chunks=(chunkSize,*self.shape), **self.kwargs)
            if self.debug: print("  done")

        self.nOutFile += 1

        if self.debug:
            with h5py.File(fName, 'r', libver='latest', swmr=True) as outFile:
                print("  created %s %dth file" % (fName, self.nOutFile), end='')
                print("  keys=", list(outFile['all_events'].keys()))
                print("  weights=", outFile['all_events/weights'].shape, end='')
                print("  shape=", outFile['all_events/images'].shape)

###################################################################################################

## Find root files with corresponding trees
print("@@@ Checking input files... (total %d files)" % (len(args.input)))
nEventTotal = 0
nEvent0s = [] 
srcFileNames = [] 
for x in args.input:
    for fName in glob(x): 
        if not fName.endswith('.root'): continue 
        f = uproot.open(fName) 
        if "data" not in f: continue 
        tree = f["data"] 
        if tree == None: continue

        if args.debug and nEventTotal == 0:
            print("-"*40)
            print("\t".join([str(key) for key in tree.keys()]))
            print("-"*40)

        srcFileNames.append(fName)
        nEvent0 = len(tree)
        nEvent0s.append(nEvent0)
        nEventTotal += nEvent0
nEventOutFile = min(nEventTotal, args.nevent) if args.nevent >= 0 else nEventTotal

fileOuts = FileSplitOut(nEventOutFile, args) 

print("@@@ Start processing...")
weightName = None if args.mc else "Weight"
# mc  default=True 'flag to set MC sample'
for nEvent0, srcFileName in zip(nEvent0s, srcFileNames):
    if args.debug: print("@@@ Open file", srcFileName)
    ## Open data files
    fin = uproot.open(srcFileName)
    tree = fin["data"]

    ## Load objects
    src_weights = np.ones(nEvent0) if weightName == None else tree[weightName]

    
    Energy_S_first = tree["Energy_S_first"].array()
    Energy_C_first = tree["Energy_C_first"].array()
    Energy_S_second = tree["Energy_S_second"].array()
    Energy_C_second = tree["Energy_C_second"].array()

    ## Save output
    fileOuts.addEvents(src_weights, Energy_S_first, Energy_C_first, Energy_S_second, Energy_C_second)

## save remaining events
fileOuts.flush()

print("@@@ Finished processing")
print("    Number of input files   =", len(srcFileNames))
print("    Number of input events  =", nEventTotal)
print("    Number of output files  =", fileOuts.nOutFile)
print("    Number of output events =", fileOuts.nOutEvent)

###################################################################################################

