#!/bin/bash
MODE=$1
PART=$2
# stringList=3dpeople,axyz,renderpeople,renderpeople_p27,humanalloy,thuman,thuman2
stringList=thuman2

# Use comma as separator and apply as pattern
for val in ${stringList//,/ }
do
    DATASET=$val
    echo "$DATASET START----------"
    # num_threads = 12
    # num_views = 36
    # resolution = 512
    # MODE = gen (process all subjects) | debug (only one subject)
    # PART = filename of render_list
    bash scripts/render_single.sh 12 $DATASET 36 512 $MODE $PART
    echo "$DATASET END----------"
done
