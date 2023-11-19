import os
from pathlib import Path

import cv2
import numpy as np


def undistort_fisheye(img, K, D, alpha=0):
  h, w = img.shape[:2]
    
  new_camera_matrix0, roi0 = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h)) 
  rectified_img = cv2.fisheye.undistortImage(img, K, D, Knew=new_camera_matrix0)
  
  return rectified_img

def main():
  input_dir = '/home/keilong/Desktop/realsense_testing_image_1611run/fisheye'
  output_dir_crop = '/home/keilong/Desktop/realsense_testing_image_1611run/rectified_crop'
  output_dir_uncrop = '/home/keilong/Desktop/realsense_testing_image_1611run/rectified_uncrop'
  

  left_K = np.array([ 2.8038512840463659e+02, 0., 4.0781082615615276e+02, 0.,
       2.7829172080942601e+02, 3.9322597989288766e+02, 0., 0., 1. ]).reshape((3, 3))
  left_D = np.array([ -5.3727663816142265e-03, 4.7786313634582786e-02,
       -4.2654049752462141e-02, 7.8549243080018073e-03 ])
  
  right_K = np.array([ 2.8048715197101194e+02, 0., 4.1222008992331848e+02, 0.,
       2.7883851721706293e+02, 3.9531102133947763e+02, 0., 0., 1. ]).reshape((3, 3))
  right_D = np.array([ -2.9173539237102989e-03, 4.2239887629178650e-02,
       -3.7738729304467269e-02, 6.2911268816686697e-03 ])
  
  # K = np.array([[2.8325647864248356e+02, 0, 4.1338173265438456e+02],
  #               [0, 2.8114932686815399e+02, 3.9173699544455792e+02],
  #               [0, 0, 1]])

  # D = np.array([ -1.2297211856003311e-02, 5.5005189166060731e-02,
  #     -4.7602777750879892e-02, 8.4774962087919032e-03 ])
  
  os.makedirs(output_dir_crop, exist_ok=True)
  os.makedirs(output_dir_uncrop, exist_ok=True)
  
  for file in os.listdir(input_dir):    
    image_name = Path(file).stem
    
    if image_name[-1:] == '1':
      K = left_K
      D = left_D
    elif image_name[-1:] == '2':
      K = right_K
      D = right_D
    
    image_path = os.path.join(input_dir, file)    
    img = cv2.imread(image_path)
    
    rectified_img_crop = undistort_fisheye(img, K, D, 0)    
    rectified_img_uncrop = undistort_fisheye(img, K, D, 1)
    
    cv2.imwrite(os.path.join(output_dir_crop, image_name + '.jpg'), rectified_img_crop)
    cv2.imwrite(os.path.join(output_dir_uncrop, image_name + '.jpg'), rectified_img_uncrop)
    
    
    
if __name__ == '__main__':
  main()