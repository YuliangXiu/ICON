#!/bin/bash
NUM_THREADS=$1
DATASET=$2
NUM_VIEWS=$3
SIZE=$4
MODE=$5
PART=$6

PYTHON_SCRIPT="./render_single.py"

if [[ $MODE == "gen" ]]; then
    echo "processing all the subjects"
    # render all the subjects
    LOG_FILE="../log/render/${DATASET}-${NUM_VIEWS}-${SIZE}-${PART}.txt"
    SAVE_DIR="../data/${DATASET}_${NUM_VIEWS}views"
    mkdir -p $SAVE_DIR
    mkdir -p "../log/render/"
    cat ../data/$DATASET/$PART.txt | shuf | xargs -P$NUM_THREADS -I {} python $PYTHON_SCRIPT -s {} -o $SAVE_DIR -r $NUM_VIEWS -w $SIZE> $LOG_FILE
fi

if [[ $MODE == "debug" ]]; then
    echo "Debug renderer"
    # render only one subject
    SAVE_DIR="../debug/${DATASET}_${NUM_VIEWS}views"
    mkdir -p $SAVE_DIR

    if [[ $DATASET == "thuman2" ]]; then
        SUBJECT="0300"
        echo "Rendering $DATASET $SUBJECT"
    fi

    python $PYTHON_SCRIPT -s $SUBJECT -o $SAVE_DIR -r $NUM_VIEWS -w $SIZE
fi
