#!/bin/sh

# parse command line arguments
while getopts s:e:m:g:l:i: flag
do
    case "${flag}" in
        s) pop_size=${OPTARG};;
        e) num_epochs=${OPTARG};;
        m) num_matches=${OPTARG};;
        g) life_gain=${OPTARG};;
        l) layers_size=${OPTARG};;
        i) initial_length=${OPTARG};;
    esac
done

# create output directories if needed
mkdir -p output
mkdir -p saved_models

# activate environment and install requirements
. venv/bin/activate > output/output.log 2> output/error.log
pip install -r requirements.txt >> output/output.log 2>> output/error.log

# disable pygame annoying message
export PYGAME_HIDE_SUPPORT_PROMPT="hide"

# train
out="output/${layers_size} 3_s${pop_size}_e${num_epochs}_m${num_matches}_g${life_gain}_i${initial_length}.log"

python3 train.py -s $pop_size -e $num_epochs -m $num_matches -g $life_gain \
-l $layers_size -i $initial_length >> ${out} 2>> output/error.log
