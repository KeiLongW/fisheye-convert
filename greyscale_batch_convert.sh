#!/bin/bash

# Set the input directory
input_dir="/raid/home/keilongwong/repositories/drone-racing-dataset/data/piloted"

# Set the output directory
output_dir="/raid/home/keilongwong/repositories/drone-racing-greyscale-dataset"

for file_path in "$input_dir"/*; do
    if [ -d "$file_path" ]; then
        # Extract the file name from the path
        file_name=$(basename "$file_path")

        # Execute the command with the current file
        python greyscale_convert.py --input_path "$file_path" --output_path "$output_dir/$file_name" --label_dir_prefix label --image_dir_prefix image
    fi
done
