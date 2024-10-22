#!/bin/bash

RECORD=2996
WORKDIR=work_dir/$RECORD
MODELNAME=runs/$RECORD

CONFIG=./config/uav-cross-subjectv1/test.yaml

WEIGHTS=runs/2101-73-27857.pt


BATCH_SIZE=16

python main.py  --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0  --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --weights $WEIGHTS
