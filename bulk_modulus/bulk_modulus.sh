#!/bin/bash
#PJM -L "rscunit=ito-a"
#PJM -L "rscgrp=ito-ss"
#PJM -L "vnode=1"
#PJM -L "vnode-core=36"
#PJM -L "elapse=0:15:00"
#PJM -j
#PJM -S

outfile=bulk_modulus.log

LANG=C

NUM_NODES=1 #${PJM_VNODES}
NUM_CORES=36
NUM_THREADS=36
NUM_PROCS=`expr ${NUM_NODES} "*" ${NUM_CORES} / ${NUM_THREADS}`
echo NUM_NODES=${NUM_NODES} NUM_CORES=${NUM_CORES} NUM_PROCS=${NUM_PROCS} NUM_THREADS=${NUM_THREADS}

export MKL_NUM_THREADS=${NUM_THREADS}
export OMP_NUM_THREADS=${NUM_THREADS},1
export OMP_STACKSIZE=64G
#ulimit -s unlimited

. ${HOME}/miniconda3/etc/profile.d/conda.sh

conda activate py38-matsci

time python ./bulk_modulus.py >& ${outfile}

conda deactivate
