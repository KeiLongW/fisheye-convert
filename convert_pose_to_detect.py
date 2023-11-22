import os
from pathlib import Path
import shutil
from arg_parser import parse_args
import time

def main():
  args = parse_args()
  
  input_dir = args.input_path
  output_dir = args.output_path
  label_dir_prefix = args.label_dir_prefix
  image_dir_prefix = args.image_dir_prefix
  
  files_in_input_dir = os.listdir(input_dir)
  label_dir = next((s for s in files_in_input_dir if s.startswith(label_dir_prefix) and os.path.isdir(os.path.join(input_dir, s))), None)
  img_dir = next((s for s in files_in_input_dir if s.startswith(image_dir_prefix) and os.path.isdir(os.path.join(input_dir, s))), None)
  os.makedirs(os.path.join(output_dir, label_dir), exist_ok=True)
  os.makedirs(os.path.join(output_dir, img_dir), exist_ok=True)
    
  print('-'*50)
  print(f'Start converting from {input_dir} to {output_dir}')
  
  start_time = time.time()
  
  for idx, file in enumerate(os.listdir(os.path.join(input_dir, label_dir))):
    if not file.endswith(".txt"):
      continue
    if os.stat(os.path.join(input_dir, label_dir, file)).st_size == 0:
      continue
    sample_name = Path(file).stem
    input_label_file = os.path.join(input_dir, label_dir, sample_name) + ".txt"
    input_img_file = os.path.join(input_dir, img_dir, sample_name) + ".jpg"
    output_label_file = os.path.join(output_dir, label_dir, sample_name) + ".txt"
    output_img_file = os.path.join(output_dir, img_dir, sample_name) + ".jpg"
    
    with open(input_label_file, 'r') as f:
      lines = f.readlines()
    with open(output_label_file, 'w') as f:
      for line_idx, line in enumerate(lines):
        new_line = ' '.join(line.split(' ')[:5]) + ('\n' if line_idx+1 < len(lines) else '')
        f.write(new_line)
    
    shutil.copy(input_img_file, output_img_file)
    
    print(f'finished {str(idx+1)} samples after time(s):', time.time() - start_time)
  

if __name__ == '__main__':
  main()