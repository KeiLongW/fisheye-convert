import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from numba import jit, njit

from arg_parser import parse_args


class PencilConvert():
  def __init__(self, input_path, 
               output_path, 
               label_dir_prefix, 
               image_dir_prefix,
               dilation_shape, 
               dilatation_size):
    self.input_path = input_path
    self.output_path = output_path
    self.label_dir_prefix = label_dir_prefix
    self.image_dir_prefix = image_dir_prefix
    self.dilation_shape = dilation_shape
    self.dilatation_size = dilatation_size
  
  def test(self):
    input_img_path = '/home/keilong/Desktop/test_defisheye/regular_imgs/images/03033_1691758682389467.jpg'
    output_path = '/home/keilong/Desktop/test_defisheye/output_pencil_test/'
    
    input_img = cv2.imread(input_img_path)
    output_img = np.zeros_like(input_img)
    
    greyscale_img = np.dot(input_img[...,:3], [0.2989, 0.5870, 0.1140])
    dilate_img = cv2.dilate(greyscale_img, np.ones((5,5), np.uint8), iterations=1)
    
    for i in range(dilate_img.shape[0]):
      for j in range(dilate_img.shape[1]):
        if dilate_img[i][j] == 0:
          output_img[i][j] = 255
        else:
          output_img[i][j] = int(255 * (greyscale_img[i][j] / dilate_img[i][j]))
    
    cv2.imwrite(output_path + 'input.jpg', input_img)
    cv2.imwrite(output_path + 'gs.jpg', greyscale_img)
    cv2.imwrite(output_path + 'dilate.jpg', dilate_img)
    cv2.imwrite(output_path + 'pencil.jpg', output_img)
    
  # https://github.com/open-airlab/pencilnet/blob/main/pencil_perception_module/nodes/pencil_filter.py
  @staticmethod
  @jit
  def _convert_one(input_img, dilation_shape, dilatation_size):
    grey_scale_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                    (dilatation_size, dilatation_size))
    dilate_img = cv2.dilate(grey_scale_img, element)
    gray_img_32_bit = grey_scale_img.copy().astype(np.uint32)
    dialted_my = ((gray_img_32_bit * 255)/dilate_img).astype(np.uint8)
    output_img = np.where(np.isnan(dialted_my), 255, dialted_my).astype(np.uint8)
    
    # dilate_img = cv2.dilate(grey_scale_img, np.ones((5,5), np.uint8), iterations=1)
    # for i in range(dilate_img.shape[0]):
    #   for j in range(dilate_img.shape[1]):
    #     if dilate_img[i][j] == 0:
    #       output_img[i][j] = 255
    #     else:
    #       output_img[i][j] = int(255 * (grey_scale_img[i][j] / dilate_img[i][j]))
    return output_img
  
    
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
      
      output_img = self._convert_one(input_img, self.dilation_shape, self.dilatation_size)
      cv2.imwrite(output_img_file, output_img)
      shutil.copy(input_label_file, output_label_file)
      
      print(f'finished {str(idx+1)} samples after time(s):', time.time() - start_time)

  def simple_convert(self):
    os.makedirs(self.output_path, exist_ok=True)
    files = os.listdir(self.input_path)
    for file in files:
      output_img_file = os.path.join(self.output_path, file)
      input_img_file = os.path.join(self.input_path, file)
      input_img = cv2.imread(input_img_file)
      output_img = self._convert_one(input_img, self.dilation_shape, self.dilatation_size)
      cv2.imwrite(output_img_file, output_img)
      
  
def main():
  args = parse_args()
  pc = PencilConvert(args.input_path, 
               args.output_path,
               args.label_dir_prefix,
               args.image_dir_prefix,
               cv2.MORPH_ELLIPSE,
               args.pencil_dilatation_size)
  pc.simple_convert()
  
if __name__ == "__main__":
  main()