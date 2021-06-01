#!/bin/bash

## Find out the job index and which file to process
I=$((${2}+1))
DATASET=${1}
[ $I -gt `cat samples/${DATASET}.txt | wc -l` ] && exit ## Skip if job index exceeds number of files to process
FILEIN=`cat samples/${DATASET}.txt | sed -n "${I}p"`

## Set basic envvars - we need cmssw FWLite
WORKDIR=$_CONDOR_JOB_IWD
cd /cvmfs/cms.cern.ch/slc7_amd64_gcc820/cms/cmssw/CMSSW_10_6_12
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval `scramv1 runtime -sh`
cd $_CONDOR_SCRATCH_DIR
OUTDIR=$WORKDIR/../data/CMSDelphes/$DATASET
if [ ! -d $OUTDIR ]; then
    mkdir -p $OUTDIR/64x64_noPU
    mkdir -p $OUTDIR/64x64_32PU
    mkdir -p $OUTDIR/224x224_noPU
    mkdir -p $OUTDIR/224x224_32PU
fi

## Print out what to do in this job.
echo "+ Job index=$I"
echo "+ CMSSW=$CMSSW_VERSION"
echo "+ FILEIN=$FILEIN"
echo "+ OUTDIR=$OUTDIR"

## Move to the Delphes directory
df -h $WORKDIR
df -h $_CONDOR_SCRATCH_DIR

## Run Delphes and do the simulation and produce images for the no-pu case
cd $WORKDIR/Delphes
./DelphesCMSFWLite $WORKDIR/cards/CMS_noPU.tcl $_CONDOR_SCRATCH_DIR/${DATASET}-noPU-${I}.root $FILEIN
echo ls -alh $_CONDOR_SCRATCH_DIR
ls -alh $_CONDOR_SCRATCH_DIR

cd $WORKDIR/atlas_dl/scripts

echo $_CONDOR_SCRATCH_DIR/${DATASET}-noPU-${I}.root > $_CONDOR_SCRATCH_DIR/filelist.txt
cat $_CONDOR_SCRATCH_DIR/filelist.txt
python ./prepare_data.py --bins  64 --input-type delphes --output-h5 $OUTDIR/64x64_noPU/hepcnn_${I}.h5 $_CONDOR_SCRATCH_DIR/filelist.txt
python ./prepare_data.py --bins 224 --input-type delphes --output-h5 $OUTDIR/224x224_noPU/hepcnn_${I}.h5 $_CONDOR_SCRATCH_DIR/filelist.txt
echo ls -alh $OUTDIR $OUTDIR/*_noPU
ls -alh $OUTDIR $OUTDIR/*_noPU

rm -f $_CONDOR_SCRATCH_DIR/*.root $_CONDOR_SCRATCH_DIR/*.h5

## Run Delphes and do the simulation and produce images for the 32-pu case
cd $WORKDIR/Delphes
./DelphesCMSFWLite $WORKDIR/cards/CMS_32PU.tcl $_CONDOR_SCRATCH_DIR/${DATASET}-32PU-${I}.root $FILEIN

cd $WORKDIR/atlas_dl/scripts

echo $_CONDOR_SCRATCH_DIR/${DATASET}-32PU-${I}.root > $_CONDOR_SCRATCH_DIR/filelist.txt
python ./prepare_data.py --bins  64 --input-type delphes --output-h5 $OUTDIR/64x64_32PU/hepcnn_${I}.h5 $_CONDOR_SCRATCH_DIR/filelist.txt
python ./prepare_data.py --bins 224 --input-type delphes --output-h5 $OUTDIR/224x224_32PU/hepcnn_${I}.h5 $_CONDOR_SCRATCH_DIR/filelist.txt

## All done. clean up the temporary files.
rm -f $_CONDOR_SCRATCH_DIR/*.root $_CONDOR_SCRATCH_DIR/*.h5
