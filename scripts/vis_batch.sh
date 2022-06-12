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
    # num_threads = 6
    # num_views = 36
    # MODE = gen (process all subjects) | debug (only one subject)
    # PART = filename of render_list
    bash scripts/vis_single.sh 6 $DATASET 36 $MODE $PART
    echo "$DATASET END----------"
done
