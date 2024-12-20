#!/bin/bash

echo whoami:$(whoami)
echo PATH:$PATH
echo PYTHONPATH:$PYTHONPATH
echo LD_LIBRARY_PATH:$LD_LIBRARY_PATH
echo LIBRARY_PATH:$LIBRARY_PATH

in_dir="/input"
tissue_dir="$in_dir/images"

echo "find -L ${in_dir} -maxdepth 2"
find -L ${in_dir} -maxdepth 2

in_path=$(find -L $in_dir -maxdepth 1 -type f)
if [ -z "$in_path" ]; then
  echo "file in /input directly not found, trying to find the file anywhere in /images"
  in_path=$(find -L $in_dir -type f ! -path "$tissue_dir/*" -print -quit)
fi

echo "input path: $in_path"
mask_path=$(find -L $tissue_dir -maxdepth 1 -type f)
if [ -z "$mask_path" ]; then
  mask_path="none"
fi
echo "mask path: $mask_path"

tmp_dir="/tmp"
#if in the input dir there is a directory named 'debug', use out_dir with _tmp suffix as tmp_dir
if [ -d "$in_dir/debug" ]; then
  tmp_dir="/output/tmp"
fi
echo "tmp dir: $tmp_dir"

if [ $# -eq 0 ]; then
	python3 /home/user/run_pack_slides.py \
		--slide_dir='/input' --spacing='/input/spacing.json' --mask_dir='/input/images' --out_dir='/output/images/packed' --mask_out_dir='/output/images/tissue-mask'
	if [ -d "$in_dir/debug" ]; then
	  chmod -R 777 /output/*
	fi
else
  echo "Executing the command: $@"
  "${@}"
fi

echo "Files in output directory:"
find -L /output

echo "Done"
