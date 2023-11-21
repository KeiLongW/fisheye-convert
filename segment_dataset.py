import os
from pathlib import Path
import matplotlib.pyplot as plt

import cv2
import numpy as np
import time 
from arg_parser import parse_args

# input_dir = '/raid/home/keilongwong/repositories/drone-racing-rectified-greyscale-dataset/flight-01a-ellipse/'
# output_dir = '/raid/home/keilongwong/repositories/drone-racing-rectified-greyscale-segmented-dataset/flight-01a-ellipse/'
# label_dir_prefix = 'label'
# image_dir_prefix = 'image'
# resize_shape = (192, 192)

# input_img_path = '/raid/home/keilongwong/repositories/drone-racing-rectified-greyscale-dataset/flight-04a-ellipse/images/01263_1691758667639905.jpg'
# input_label_path = '/raid/home/keilongwong/repositories/drone-racing-rectified-greyscale-dataset/flight-04a-ellipse/labels/01263_1691758667639905.txt'


def convert_one(img, bbs, kps, resize=None):
  bbs[:, 2] = bbs[:, 2] * (img.shape[1]-1)  # width
  bbs[:, 3] = bbs[:, 3] * (img.shape[0]-1)  # height
  bbs[:, 0] = bbs[:, 0] * (img.shape[1]-1) - bbs[:, 2]/2 # x
  bbs[:, 1] = bbs[:, 1] * (img.shape[0]-1) - bbs[:, 3]/2 # y
  
  kps[:, :, 0] = kps[:, :, 0] * (img.shape[1]-1)
  kps[:, :, 1] = kps[:, :, 1] * (img.shape[0]-1)
  
  cropped_imgs = []
  kp_groups = []
  
  for idx, bb in enumerate(bbs):
    x, y, w, h = bb
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    cropped_img = img[y:y+h, x:x+w]
    
    if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
      continue
    
    # resize cropped img
    if resize is not None:
      cropped_img = cv2.resize(cropped_img, resize)
    cropped_imgs.append(cropped_img)
    
    # write the labels of each cropped image to txt
    kp_group = kps[idx]
    for kp in kp_group:
      if kp[2] != 2:
        continue
      kp[0] = kp[0] - x
      kp[1] = kp[1] - y
    
    # resize kp as well
    if resize is not None:
      kp_group[:, 0] = kp_group[:, 0] * (resize[0]/w)
      kp_group[:, 1] = kp_group[:, 1] * (resize[1]/h)
    # scale = np.flipud(np.divide(cropped_img.shape[:2], img[y:y+h, x:x+w].shape[:2]))
    # kp_group[:, [0,1]] = np.multiply(kp_group[:, [0,1]], scale)
    kp_group[:, 0] = kp_group[:, 0] / (cropped_img.shape[1]-1)
    kp_group[:, 1] = kp_group[:, 1] / (cropped_img.shape[0]-1)
    
    for kp in kp_group:
      # if the kp is outside the image, it means it is unseen from the image, we should ignore them and set as 0
      if kp[0] < 0 or kp[0] >= 1 or kp[1] < 0 or kp[1] >= 1:
        kp[0] = 0
        kp[1] = 0
        kp[2] = 0
    
    kp_group = kp_group.reshape((1, -1))
    kp_groups.append(kp_group)
  
  return cropped_imgs, kp_groups
    
def main():
  args = parse_args()
  
  input_dir = args.input_path
  output_dir = args.output_path
  label_dir_prefix = args.label_dir_prefix
  image_dir_prefix = args.image_dir_prefix
  if args.resize_x is not None and args.resize_y is not None:
    resize_shape = (args.resize_x, args.resize_y)
  else:
    resize_shape = None
  
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
    
    input_img = cv2.imread(input_img_file, cv2.IMREAD_UNCHANGED)
    input_label = np.loadtxt(input_label_file, delimiter=" ", dtype=np.float32)
    input_label = input_label.reshape(-1, input_label.shape[-1])
  
    bbs = input_label[:, 1:5]
    kps = input_label[:, 5:].reshape((-1, 4, 3))
    cropped_imgs, kp_groups = convert_one(input_img, bbs, kps, resize=resize_shape)
    
    for cropped_idx in range(len(cropped_imgs)):
      output_label_file = os.path.join(output_dir, label_dir, sample_name) + f"_{cropped_idx}" + ".txt"
      output_img_file = os.path.join(output_dir, img_dir, sample_name) + f"_{cropped_idx}" + ".jpg"
      np.savetxt(output_label_file, kp_groups[cropped_idx], delimiter=" ", fmt="%f")
      cv2.imwrite(output_img_file, cropped_imgs[cropped_idx])
    
    print(f'finished {str(idx+1)} samples after time(s):', time.time() - start_time)
  

if __name__ == '__main__':
  main()


# img = cv2.imread(input_img_path)
# labels = np.loadtxt(input_label_path, delimiter=' ', dtype=np.float32)

# bbs = labels[:, 1:5]
# kps = labels[:, 5:].reshape((-1, 4, 3))

# bbs[:, 2] = bbs[:, 2] * (img.shape[1]-1)  # width
# bbs[:, 3] = bbs[:, 3] * (img.shape[0]-1)  # height
# bbs[:, 0] = bbs[:, 0] * (img.shape[1]-1) - bbs[:, 2]/2 # x
# bbs[:, 1] = bbs[:, 1] * (img.shape[0]-1) - bbs[:, 3]/2 # y

# kps[:, :, 0] = kps[:, :, 0] * (img.shape[1]-1)
# kps[:, :, 1] = kps[:, :, 1] * (img.shape[0]-1)

# # draw kps to original img
# keypoint_colors = [(255,255,204), (166,214,8), (0,255,255), (255,0,255)]
# draw_img = img.copy()
# for kp_group in kps:
#   for kp_idx, kp in enumerate(kp_group):
#     if kp[2] != 2:
#         continue
#     color = keypoint_colors[kp_idx]
#     # draw point to original image
#     draw_img = cv2.circle(draw_img, (int(kp[0]), int(kp[1])), 1, color, 1)
# cv2.imwrite(os.path.join(output_dir, 'draw_img.jpg'), draw_img)

# # crop img based on bbs
# for idx, bb in enumerate(bbs):
#   x, y, w, h = bb
#   x = int(x)
#   y = int(y)
#   w = int(w)
#   h = int(h)
#   cropped_img = img[y:y+h, x:x+w]
  
#   # resize cropped img
#   cropped_img = cv2.resize(cropped_img, (640, 480))
  
#   cv2.imwrite(os.path.join(output_dir, f'cropped_{idx}.jpg'), cropped_img)
  
#   # write the labels of each cropped image to txt
#   kp_group = kps[idx]
#   kp_group[:, 0] = kp_group[:, 0] - x
#   kp_group[:, 1] = kp_group[:, 1] - y
  
#   # resize kp as well
#   kp_group[:, 0] = kp_group[:, 0] * (640/w)
#   kp_group[:, 1] = kp_group[:, 1] * (480/h)
#   # scale = np.flipud(np.divide(cropped_img.shape[:2], img[y:y+h, x:x+w].shape[:2]))
#   # kp_group[:, [0,1]] = np.multiply(kp_group[:, [0,1]], scale)
  
#   kp_group[:, 0] = kp_group[:, 0] / (cropped_img.shape[1]-1)
#   kp_group[:, 1] = kp_group[:, 1] / (cropped_img.shape[0]-1)
#   kp_group = kp_group.reshape((1, -1))
#   np.savetxt(os.path.join(output_dir, f'cropped_{idx}.txt'), kp_group, delimiter=' ', fmt='%f')
  
#   # draw a labeled image of each cropped image
#   keypoint_colors = [(255,255,204), (166,214,8), (0,255,255), (255,0,255)]
#   kp_group = kp_group.reshape((4, 3))
#   kp_group[:, 0] = kp_group[:, 0] * (cropped_img.shape[1]-1)
#   kp_group[:, 1] = kp_group[:, 1] * (cropped_img.shape[0]-1)
#   labeled_cropped_img = cropped_img.copy()
#   for kp_idx, kp in enumerate(kp_group):
#     if kp[2] != 2:
#         continue
#     color = keypoint_colors[kp_idx]
#     # draw point to cropped image
#     labeled_cropped_img = cv2.circle(labeled_cropped_img, (int(kp[0]), int(kp[1])), 1, color, 1)
#   # save image
#   cv2.imwrite(os.path.join(output_dir, f'cropped_{idx}_labeled.jpg'), labeled_cropped_img)
