import os
import time
from math import atan2, cos, sin, sqrt
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from arg_parser import parse_args


@njit
def _convert(i,j,img):
    x = j - (img.shape[1]/2)
    y = ((img.shape[0] - i) - (img.shape[0]/2))
    return x,y

@njit
def _normalize(x,y,img):
    xn = x/(img.shape[1]/2)
    yn = y/(img.shape[0]/2)
    return xn,yn

@njit
def _denormalize(xn,yn,img):
    x = xn * (img.shape[1]/2)
    y = yn * (img.shape[0]/2)
    return x,y

@njit
def _de_convert(x,y,img):
    j = int(x + (img.shape[1]/2))
    i = int(img.shape[0] - (y + (img.shape[0]/2)))
    return i,j

class Fisheye():
  def __init__(self, input_path, 
               output_path, 
               label_dir_prefix, 
               image_dir_prefix,
               fisheye_bb_convert):
    self.input_path = input_path
    self.output_path = output_path
    self.label_dir_prefix = label_dir_prefix
    self.image_dir_prefix = image_dir_prefix
    self.fisheye_bb_convert = fisheye_bb_convert

  @staticmethod
  @njit
  def _convert_one(input_img, input_bbs, input_kps, fisheye_bb_convert):
    # start_time = time.time()
    
    # input_bbs = input_label[:, 1:5]
    input_bbs[:, 2] = input_bbs[:, 2] * input_img.shape[1]
    input_bbs[:, 3] = input_bbs[:, 3] * input_img.shape[0]
    input_bbs[:, 0] = input_bbs[:, 0] * input_img.shape[1] - input_bbs[:, 2]/2
    input_bbs[:, 1] = input_bbs[:, 1] * input_img.shape[0] - input_bbs[:, 3]/2
    # input_bbs = input_bbs.astype(np.uint8)
    
    # input_kps = input_label[:, 5:].reshape((-1, 4, 3))
    input_kps[:, :, 0] = input_kps[:, :, 0] * input_img.shape[1]
    input_kps[:, :, 1] = input_kps[:, :, 1] * input_img.shape[0]
    # input_kps = input_kps.astype(np.uint8)
    
    output_bbs = np.zeros(input_bbs.shape, np.float32)
    output_kps = np.zeros(input_kps.shape, np.float32)    
    output_img = np.zeros((input_img.shape[0], input_img.shape[1], 3), np.uint8)
    
    test_img = np.full((input_img.shape[0], input_img.shape[1], 3), 255)
    
    # print('finished preprocessing:', time.time() - start_time)
    
    for i in range(input_img.shape[0]):
      for j in range(input_img.shape[1]):
        x, y = _convert(i, j, input_img)
        xn, yn = _normalize(x, y, input_img)
        r = sqrt((xn**2) + (yn**2))
        theta = atan2(yn, xn)
        if (r <= 1):
          r_prime = (r + 1 - sqrt(1 - r**2)) / 2
          xn_new = r_prime * cos(theta)
          yn_new = r_prime * sin(theta)
          x_new, y_new = _denormalize(xn_new, yn_new, input_img)
          new_i, new_j = _de_convert(x_new, y_new, input_img)
          
          output_img[i][j][0] = input_img[new_i][new_j][0]
          output_img[i][j][1] = input_img[new_i][new_j][1]
          output_img[i][j][2] = input_img[new_i][new_j][2]
          
          test_img[new_i][new_j][0] = input_img[new_i][new_j][0]
          test_img[new_i][new_j][1] = input_img[new_i][new_j][1]
          test_img[new_i][new_j][2] = input_img[new_i][new_j][2]
          
          for idx, kp_group in enumerate(input_kps):
            for idy, kp in enumerate(kp_group):
              if kp[2] == 0:
                continue
              if new_j-1 < kp[0] < new_j+1 and new_i-1 < kp[1] < new_i+1:
                output_kps[idx][idy][0] = j
                output_kps[idx][idy][1] = i
                output_kps[idx][idy][2] = 2
                
          if fisheye_bb_convert == "pix_to_pix":          
            for idx, bb in enumerate(input_bbs):
              if new_j-1 < bb[0] < new_j+1:
                output_bbs[idx][0] = j
              if new_i-1 < bb[1] < new_i+1:
                output_bbs[idx][1] = i            
              if new_j-1 < bb[2] < new_j+1:
                output_bbs[idx][2] = j
              if new_i-1 < bb[3] < new_i+1:
                output_bbs[idx][3] = i
                
    
    # print('finished converting:', time.time() - start_time)
    
    if fisheye_bb_convert == "outside_kp":
      for idx, bb in enumerate(input_bbs):
        output_kp_wo_zero = output_kps[idx][np.where(output_kps[idx][:,2]!=0)]
        left_most = output_kp_wo_zero[:, 0].min()
        right_most = output_kp_wo_zero[:, 0].max()
        top_most = output_kp_wo_zero[:, 1].min()
        bottom_most = output_kp_wo_zero[:, 1].max()
        
        horizontal_padding = (right_most - left_most) * 0.25
        vertical_padding = (bottom_most - top_most) * 0.25
        
        output_bbs[idx][0] = left_most - horizontal_padding
        output_bbs[idx][1] = top_most - vertical_padding
        output_bbs[idx][2] = right_most - left_most + horizontal_padding*2
        output_bbs[idx][3] = bottom_most - top_most + vertical_padding*2
        
        # avoiding out of image size
        output_bbs[idx][0] = output_bbs[idx][0] if output_bbs[idx][0] > 0 else 0
        output_bbs[idx][1] = output_bbs[idx][1] if output_bbs[idx][1] > 0 else 0
        output_bbs[idx][2] = output_bbs[idx][2] if output_bbs[idx][2] + output_bbs[idx][0] < output_img.shape[1] else output_img.shape[1] - 1 - output_bbs[idx][0]
        output_bbs[idx][3] = output_bbs[idx][3] if output_bbs[idx][3] + output_bbs[idx][1] < output_img.shape[0] else output_img.shape[0] - 1 - output_bbs[idx][1]

    output_bbs[:, 0] = (output_bbs[:, 0] + output_bbs[:, 2]/2) / output_img.shape[1]
    output_bbs[:, 1] = (output_bbs[:, 1] + output_bbs[:, 3]/2) / output_img.shape[0]
    output_bbs[:, 2] = output_bbs[:, 2] / output_img.shape[1]
    output_bbs[:, 3] = output_bbs[:, 3] / output_img.shape[0]
    output_kps[:, :, 0] = output_kps[:, :, 0] / output_img.shape[1]
    output_kps[:, :, 1] = output_kps[:, :, 1] / output_img.shape[0]
    # output_label = np.concatenate((input_label[:, [0]], output_bbs, output_kps.reshape((-1, 12))), axis=1)
    
    # print('finished postprocessing:', time.time() - start_time)
    
    return output_img, output_bbs, output_kps
    
    
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
      input_label = np.loadtxt(input_label_file, delimiter=" ", dtype=np.float32)
      input_label = input_label.reshape(-1, input_label.shape[-1])
      
      input_bbs = input_label[:, 1:5]
      input_kps = input_label[:, 5:].reshape((-1, 4, 3))
      
      output_img, output_bbs, output_kps = self._convert_one(input_img, input_bbs, input_kps, self.fisheye_bb_convert)
      output_label = np.concatenate((input_label[:, [0]], output_bbs, output_kps.reshape((-1, 12))), axis=1)
      np.savetxt(output_label_file, output_label, delimiter=" ", fmt="%f")
      cv2.imwrite(output_img_file, output_img)
      
      print(f'finished {str(idx+1)} samples after time(s):', time.time() - start_time)
    
def main():
  args = parse_args()
  fe = Fisheye(args.input_path, 
               args.output_path,
               args.label_dir_prefix,
               args.image_dir_prefix,
               args.fisheye_bb_convert)
  fe.convert()

if __name__ == '__main__':
  main()