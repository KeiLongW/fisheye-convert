import os
import time
from pathlib import Path
import numpy as np
from numba import njit
import cv2
import shutil
from arg_parser import parse_args

class GreyscaleConvert():
  def __init__(self, input_path, 
               output_path, 
               label_dir_prefix, 
               image_dir_prefix):
    self.input_path = input_path
    self.output_path = output_path
    self.label_dir_prefix = label_dir_prefix
    self.image_dir_prefix = image_dir_prefix
    
  @staticmethod
  # @njit
  def _convert_one(input_img):
    return np.dot(input_img[...,:3], [0.2989, 0.5870, 0.1140])
    
  def convert(self):
    files = os.listdir(self.input_path)
    label_dir = next((s for s in files if s.startswith(self.label_dir_prefix) and os.path.isdir(os.path.join(self.input_path, s))), None)
    img_dir = next((s for s in files if s.startswith(self.image_dir_prefix) and os.path.isdir(os.path.join(self.input_path, s))), None)
    os.makedirs(os.path.join(self.output_path, label_dir), exist_ok=True)
    os.makedirs(os.path.join(self.output_path, img_dir), exist_ok=True)
    
    print('-'*50)
    print(f'Start converting from {self.input_path} to {self.output_path}')
    
    start_time = time.time()
    for idx, file in enumerate(os.listdir(os.path.join(self.input_path, label_dir))):
      if not file.endswith(".txt"):
        continue
      sample_name = Path(file).stem
      input_label_file = os.path.join(self.input_path, label_dir, sample_name) + ".txt"
      input_img_file = os.path.join(self.input_path, img_dir, sample_name) + ".jpg"      
      output_label_file = os.path.join(self.output_path, label_dir, sample_name) + ".txt"
      output_img_file = os.path.join(self.output_path, img_dir, sample_name) + ".jpg"
      
      input_img = cv2.imread(input_img_file, cv2.IMREAD_UNCHANGED)
      
      output_img = self._convert_one(input_img)
      cv2.imwrite(output_img_file, output_img)
      shutil.copy(input_label_file, output_label_file)
      
      print(f'finished {str(idx+1)} samples after time(s):', time.time() - start_time)
      

def main():
  args = parse_args()
  gs = GreyscaleConvert(args.input_path, 
               args.output_path,
               args.label_dir_prefix,
               args.image_dir_prefix)
  gs.convert()

if __name__ == '__main__':
  main()