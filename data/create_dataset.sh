#!/bin/sh
# Create (render) the dataset from the source .blend file. This takes a while.

blender $1 --background -noaudio --python data/create_dataset.py -- $2 2>/dev/null