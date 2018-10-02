from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
from math import ceil
import os
import pickle
import h5py
import numpy as np
import shutil
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import random

from models.keras_ssd300 import ssd_300
from models.keras_mobileNet_ssd import mobilenet_ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

img_height = 300                                            # Height of the model input images
img_width = 300                                             # Width of the model input images
img_channels = 3                                            # Number of color channels of the model input images
mean_color = [123, 117, 104]                                # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0]                                   # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 20                                              # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]    # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0]]                           # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300]                           # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]                    # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False                                          # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2]                            # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True

K.clear_session() # Clear previous models from memory.

model = mobilenet_ssd_300( image_size=(img_height, img_width, img_channels),
                                       n_classes=n_classes,
                                       mode='inference',
                                       l2_regularization=0.0005,
                                       scales=scales,
                                       aspect_ratios_per_layer=aspect_ratios,
                                       two_boxes_for_ar1=two_boxes_for_ar1,
                                       steps=None,
                                       offsets=offsets,
                                       clip_boxes=clip_boxes,
                                       variances=variances,
                                       normalize_coords=normalize_coords,
                                       subtract_mean=mean_color,
                                       swap_channels=swap_channels)

model_path = 'mn_ssd300_train_models_fd1/mn_ssd300_face_detection_epoch-118_loss-4.4522_val_loss-4.1524.h5'
model.load_weights(model_path, by_name=True)

WIDER_test_images_dir = 'dataset/WIDER_test/images/'
WIDER_test_filename = 'dataset/wider_face_split/wider_face_test_filelist.txt'

test_filenames = []

with open(WIDER_test_filename) as f:
    test_filenames = [line.rstrip() for line in f]

for num in range(6):
    input_images = [] # Store resized versions of the images here.
    row_images = []
    image_paths = []

    random_index = random.randint(0, len(test_filenames) - 1)

    row_images.append(imread(WIDER_test_images_dir + test_filenames[random_index]))
    image_paths.append(WIDER_test_images_dir + test_filenames[random_index])

    img = image.load_img(image_paths[0], target_size=(img_height, img_width))
    img = image.img_to_array(img)
    input_images.append(img)
    input_images = np.array(input_images)

    y_pred = model.predict(input_images, batch_size=8)

    confidence_threshold = 0.5

    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_thresh[0])

    colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()

    plt.figure(figsize=(20,12))
    plt.imshow(row_images[0])

    current_axis = plt.gca()

    for box in y_pred_thresh[0]:
        xmin = box[-4] * row_images[0].shape[1] / img_width
        ymin = box[-3] * row_images[0].shape[0] / img_height
        xmax = box[-2] * row_images[0].shape[1] / img_width
        ymax = box[-1] * row_images[0].shape[0] / img_height
        color = colors[int(box[0])]
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  

    output_image_dir = 'face_detection_images'

    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    plt.savefig(os.path.join(output_image_dir, 'result_{}_image{}.png'.format(num, random_index)))
