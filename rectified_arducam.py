import cv2
import numpy as np


def get_output(x, y, counter):
  output_img = np.zeros_like(input_img)
  output_img.fill(255)
  output_img[x][y] = counter
  cv2.imwrite(output_path.format(counter), output_img)
  return output_img

image_path = '/home/keilong/repositories/drone-racing-dataset/data/autonomous/flight-02a-ellipse/camera_flight-02a-ellipse/01578_1691757122212387.jpg'
label_path = '/home/keilong/repositories/drone-racing-dataset/data/autonomous/flight-02a-ellipse/label_flight-02a-ellipse/01578_1691757122212387.txt'

output_path = '/home/keilong/Desktop/test_arducam_rectify/01578_1691757122212387_{0}.png'


input_img = cv2.imread(image_path)
input_label = np.loadtxt(label_path, delimiter=" ", dtype=np.float32)
input_label = input_label.reshape(-1, input_label.shape[-1])

input_bbs = input_label[:, 1:5]
input_kps = input_label[:, 5:].reshape((-1, 4, 3))

input_bbs[:, 2] = (input_bbs[:, 2] * input_img.shape[1]).astype(int)
input_bbs[:, 3] = (input_bbs[:, 3] * input_img.shape[0]).astype(int)
input_bbs[:, 0] = (input_bbs[:, 0] * input_img.shape[1] - input_bbs[:, 2]/2).astype(int)
input_bbs[:, 1] = (input_bbs[:, 1] * input_img.shape[0] - input_bbs[:, 3]/2).astype(int)

input_kps[:, :, 0] = (input_kps[:, :, 0] * input_img.shape[1]).astype(int)
input_kps[:, :, 1] = (input_kps[:, :, 1] * input_img.shape[0]).astype(int)

input_bbs = input_bbs.astype(int)
input_kps = input_kps.astype(int)

output_img = np.zeros_like(input_img)
output_img.fill(255)

# output_img = input_img

counter = 0
for idx in range(len(input_bbs)):
  bb = input_bbs[idx]
  kps = input_kps[idx]
  
  # output_img[bb[1]][bb[0]] = counter
  output_img = get_output(bb[1], bb[0], counter)
  counter += 1
  # output_img[bb[1]][bb[0] + bb[2]] = counter
  output_img = get_output(bb[1], bb[0] + bb[2], counter)
  counter += 1
  # output_img[bb[1] + bb[3]][bb[0]] = counter
  output_img = get_output(bb[1] + bb[3], bb[0], counter)
  counter += 1
  
  for kp in kps:
    if kp[2] == 2:
      # output_img[kp[1]][kp[0]] = counter
      output_img = get_output(kp[1], kp[0], counter)
    counter += 1
