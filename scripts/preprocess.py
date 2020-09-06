#!/usr/bin/env python3
import h5py
import numpy as np
import argparse
import sys
from math import ceil

if sys.version_info[0] < 3: sys.exit()

parser = argparse.ArgumentParser()
parser.add_argument('input', nargs='+', action='store', type=str, help='input file names')
parser.add_argument('-o', '--output', action='store', type=str, help='output file name', required=True)
parser.add_argument('--nevent', action='store', type=int, default=-1, help='number of events to preprocess')
parser.add_argument('--nfiles', action='store', type=int, default=0, help='number of output files')
parser.add_argument('--format', action='store', choices=('NHWC', 'NCHW'), default='NCHW', help='image format for output (NHWC for TF default, NCHW for pytorch default)')
parser.add_argument('-c', '--chunk', action='store', type=int, default=1024, help='chunk size')
parser.add_argument('--compress', action='store', choices=('gzip', 'lzf', 'none'), default='none', help='compression algorithm')
parser.add_argument('-s', '--split', action='store_true', default=False, help='split output file')
parser.add_argument('-d', '--debug', action='store_true', default=False, help='debugging')
parser.add_argument('--precision', action='store', type=int, choices=(8,16,32,64), default=32, help='Precision')
parser.add_argument('--dotrackpt', action='store_true', default=False, help='Choose track pt for the 3rd channel, rather than the track counts')
args = parser.parse_args()

srcFileNames = [x for x in args.input if x.endswith('.h5')]
if not args.output.endswith('.h5'): outPrefix, outSuffix = args.output+'/data', '.h5'
else: outPrefix, outSuffix = args.output.rsplit('.', 1)
args.nevent = max(args.nevent, -1) ## nevent should be -1 to process everything or give specific value

## Logic for the arguments regarding on splitting
##   split off: we will simply ignore nfiles parameter => reset nfiles=1
##     nevent == -1: process all events store in one file
##     nevent != -1: process portion of events, store in one file
##   split on:
##     nevent == -1, nfiles == 1: same as the one without splitting
##     nevent != -1, nfiles == 1: same as the one without splitting
##     nevent == -1, nfiles != 1: process all events, split into nfiles
##     nevent != -1, nfiles != 1: split files, limit total number of events to be nevent
##     nevent != -1, nfiles == 0: split files by nevents for each files
if not args.split or args.nfiles == 1:
    ## Just confirm the options for no-splitting case
    args.split = False
    args.nfiles = 1
elif args.split and args.nevent > 0:
    args.nfiles = 0

## First scan files to get total number of events
print("@@@ Checking input files... (total %d files)" % (len(args.input)))
nEventTotal = 0
nEvent0s = []
for srcFileName in srcFileNames:
    data = h5py.File(srcFileName, 'r')['all_events']
    nEvent0 = data['hist'].shape[0]
    nEvent0s.append(nEvent0)
    nEventTotal += nEvent0
if args.nfiles > 0:
    nEventOutFile = int(ceil(nEventTotal/args.nfiles))
else:
    args.nfiles = int(ceil(nEventTotal/args.nevent))
    nEventOutFile = min(nEventTotal, args.nevent)

shape = h5py.File(srcFileName, 'r')['all_events/hist'].shape[1:]
shape = [3,*shape] if args.format == 'NCHW' else [*shape,3]

print("@@@ Total %d events to process, store into %d files (%d events per file)" % (nEventTotal, args.nfiles, nEventOutFile))

class FileSplitOut:
    def __init__(self, shape, maxEvent, fNamePrefix, args, nEventTotal, debug=False):
        self.shape = shape
        self.maxEvent = maxEvent
        self.fNamePrefix = fNamePrefix
        self.chunkSize = args.chunk
        self.nEventTotal = nEventTotal
        self.debug = debug
        self.chAxis = -1 if args.format == 'NHWC' else 1

        precision = 'f%d' % (args.precision//8)
        self.kwargs = {'dtype':precision}
        if args.compress == 'gzip':
            self.kwargs.update({'compression':'gzip', 'compression_opts':9})
        elif args.compress == 'lzf':
            self.kwargs.update({'compression':'lzf'})

        self.nOutFile = 0
        self.nOutEvent = 0

        self.initOutput()

    def initOutput(self):
        ## Build placeholder for the output
        self.labels = np.ones(0)
        self.weights = np.ones(0)
        self.images = np.ones([0,*self.shape])

    def addEvents(self, src_labels, src_weights, src_images_h, src_images_e, src_images_t):

        nSrcEvent = len(src_weights)
        #self.nOutEvent += nSrcEvent;
        begin = 0
        while begin < nSrcEvent:
            end = begin+min(self.maxEvent, nSrcEvent)
            self.nOutEvent += (end-begin)
            print("%d/%d" % (self.nOutEvent, self.nEventTotal), end='\r')

            images_h = np.expand_dims(src_images_h[begin:end,:,:], self.chAxis)
            images_e = np.expand_dims(src_images_e[begin:end,:,:], self.chAxis)
            images_t = np.expand_dims(src_images_t[begin:end,:,:], self.chAxis)
            images = np.concatenate([images_h, images_e, images_t], axis=self.chAxis)

            self.labels  = np.concatenate([self.labels , src_labels[begin:end]])
            self.weights = np.concatenate([self.weights, src_weights[begin:end]])
            self.images  = np.concatenate([self.images , images])

            if len(self.weights) == self.maxEvent: self.flush()
            begin = end

    def flush(self):
        self.save()
        self.initOutput()

    def save(self):
        fName = "%s_%d.h5" % (self.fNamePrefix, self.nOutFile)
        nEventToSave = len(self.weights)
        if nEventToSave == 0: return
        if self.debug: print("Writing output file %s..." % fName, end='')

        chunkSize = min(self.chunkSize, nEventToSave)
        with h5py.File(fName, 'w', libver='latest', swmr=True) as outFile:
            g = outFile.create_group('all_events')
            g.create_dataset('labels' , data=self.labels , chunks=(chunkSize,), dtype='f4')
            g.create_dataset('weights', data=self.weights, chunks=(chunkSize,), dtype='f4')
            g.create_dataset('images' , data=self.images , chunks=((chunkSize,)+self.images.shape[1:]), **self.kwargs)
            if self.debug: print("  done")

        self.nOutFile += 1

        if self.debug:
            with h5py.File(fName, 'r', libver='latest', swmr=True) as outFile:
                print("  created %s %dth file" % (fName, self.nOutFile), end='')
                print("  keys=", list(outFile.keys()), end='')
                print("  shape=", outFile['all_events']['images'].shape)

print("@@@ Start processing...")
fileOuts = FileSplitOut(shape, nEventOutFile, outPrefix, args, nEventTotal, args.debug)

outFileNames = []
for nEvent0, srcFileName in zip(nEvent0s, srcFileNames):
    if args.debug: print("Open file", srcFileName)
    ## Open data file
    data = h5py.File(srcFileName, 'r')['all_events']

    weights = data['weight']
    labels = data['y'] if 'y' in data else np.ones(weights.shape[0])

    image_h = data['hist']
    image_e = data['histEM']
    image_t = data['histtrack' if not args.dotrackpt else 'histtrackPt']

    ## Preprocess image
    #image_e /= np.max(image_e)
    #image_t /= np.max(image_t)

    ## Put into the output file
    fileOuts.addEvents(labels, weights, image_h, image_e, image_t)

## save remaining events
fileOuts.flush()

print("@@@ Finished processing")
print("    Number of input files   =", len(srcFileNames))
print("    Number of input events  =", nEventTotal)
print("    Number of output files  =", fileOuts.nOutFile)
print("    Number of output events =", fileOuts.nOutEvent)

