#!/bin/bash

#PBS -V
#PBS -N torch_hepcnn
#PBS -q normal
#PBS -W sandbox=PRIVATE
#PBS -A etc
#PBS -l select=1:ncpus=68:mpiprocs=1:ompthreads=64
#PBS -l walltime=01:00:00

source /apps/applications/miniconda3/etc/profile.d/conda.sh
#module load gcc/7.2.0 openmpi/3.1.0 craype-mic-knl tensorflow/1.12.0 hdf5-parallel/1.10.2
source /apps/compiler/intel/19.0.5/impi/2019.5.281/intel64/bin/mpivars.sh release_mt
#conda activate pytorch_v1.1.0
module load git craype-mic-knl
module load gcc/8.3.0
export USE_CUDA=0
export USE_MKLDNN=1
export USE_OPENMP=1
export USE_TBB=0
conda activate /scratch/$USER/conda/nurion_torch

export HDF5_USE_FILE_LOCKING='FALSE'
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export KMP_AFFINITY=granularity=fine,compact
#export KMP_AFFINITY=granularity=fine,scatter
#export KMP_AFFINITY=granularity=fine,balanced
export KMP_SETTINGS=1
export CUDA_VISIBLE_DEVICES=""

export OMPI_MCA_btl_openib_allow_ib=1
export OMPI_MCA_btl_openib_if_include="hfi1_0:1"
export LD_LIBRARY_PATH=/opt/pbs/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=64

[ _$BATCH == _ ] && BATCH=32
[ _$SELECT == _ ] && SELECT=50
[ _$EPOCH == _ ] && EPOCH=50
[ _$KMP_BLOCKTIME == _ ] && export KMP_BLOCKTIME=200
[ _$MODEL == _ ] && MODEL=default
[ _$SAMPLEDIR == _ ] && export SAMPLEDIR=/scratch/x1797a01/data/hdf5_noPU_64x64
OUTDIR=hepcnn_${MODEL}_32PU_64x64/SELECT_${SELECT}__BATCH_${BATCH}

[ _$PBS_O_WORKDIR != _ ] && cd $PBS_O_WORKDIR
[ -d $OUTDIR ] || mkdir -p $OUTDIR
mpirun -np $SELECT -env OMP_NUM_THREADS $OMP_NUM_THREADS \
    python train_labelByUser.py -o $OUTDIR \
           --epoch $EPOCH --batch $BATCH \

