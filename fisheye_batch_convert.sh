#!/bin/bash

# Set the input directory
input_dir="/workspace/drone-racing-dataset/data/autonomous"

# Set the output directory
output_dir="/workspace/drone-racing-fisheye-dataset"

for file_path in "$input_dir"/*; do
    if [ -d "$file_path" ]; then
        # Extract the file name from the path
        file_name=$(basename "$file_path")

        # Execute the command with the current file
        python fisheye_convert.py --input_path "$file_path" --output_path "$output_dir/$file_name" --label_dir_prefix label --image_dir_prefix image
    fi
done
