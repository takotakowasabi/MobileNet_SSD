# coding: utf-8
from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from scipy.misc import imread
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import os

from models.keras_ssd300 import ssd_300
from models.keras_mobileNet_ssd_ReLU6 import mobilenet_ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator

import sys
import csv

# Set a few configuration parameters.
img_height = 300
img_width = 300
model_mode = 'inference'

# Set the path of the trained weights.
weights_path = 'weights_for_evaluation/ssd300_pascal_07+12_epoch-116_loss-4.5341_val_loss-4.2570.h5'

# Set the paths to the dataset here.
dataset_images_dir = 'dataset/WIDER_val/images'
dataset_annotations_dir = 'dataset/WIDER_val/annotations'
test_image_set_filename = 'wider_face_test_filelist_small.txt'

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background', 'face']
n_classes = len(classes)-1

def main():
    # create dataset
    dataset = DataGenerator()
    dataset.parse_xml(images_dirs=[dataset_images_dir],
                      image_set_filenames=[test_image_set_filename],
                      annotations_dirs=[dataset_annotations_dir],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=False,
                      ret=False)
    
    # create model
    model = ssd_300(image_size=(img_height, img_width, 3),
                              n_classes=n_classes,
                              mode=model_mode,
                              l2_regularization=0.0005,
                              scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                              aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                       [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                       [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                       [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                       [1.0, 2.0, 0.5],
                                                       [1.0, 2.0, 0.5]],
                              two_boxes_for_ar1=True,
                              steps=None,
                              offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                              clip_boxes=False,
                              variances=[0.1, 0.1, 0.2, 0.2],
                              normalize_coords=True,
                              subtract_mean=[123, 117, 104],
                              swap_channels=[2, 1, 0],
                              confidence_thresh=1.0e-4,
                              iou_threshold=0.45,
                              top_k=200,
                              nms_max_output_size=400)

    # load weights and compile it
    model.load_weights(weights_path, by_name=True)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    evaluator = Evaluator(model=model,
                          n_classes=n_classes,
                          data_generator=dataset,
                          model_mode=model_mode)

    results = evaluator(img_height=img_height,
                        img_width=img_width,
                        batch_size=8,
                        data_generator_mode='resize',
                        round_confidences=False,
                        matching_iou_threshold=0.5,
                        border_pixels='include',
                        sorting_algorithm='quicksort',
                        average_precision_mode='sample',
                        num_recall_points=11,
                        ignore_neutral_boxes=True,
                        return_precisions=True,
                        return_recalls=True,
                        return_average_precisions=True,
                        verbose=True)

    mean_average_precision, average_precisions, precisions, recalls = results

    for i in range(1, len(average_precisions)):
      print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
    print()
    print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))

    m = max((n_classes + 1) // 2, 2)
    n = 2

    fig, cells = plt.subplots(m, n, figsize=(n*8,m*8))
    for i in range(m):
        for j in range(n):
            if n*i+j+1 > n_classes: break
            cells[i, j].plot(recalls[n*i+j+1], precisions[n*i+j+1], color='blue', linewidth=1.0)
            cells[i, j].set_xlabel('recall', fontsize=14)
            cells[i, j].set_ylabel('precision', fontsize=14)
            cells[i, j].grid(True)
            cells[i, j].set_xticks(np.linspace(0,1,6))
            cells[i, j].set_yticks(np.linspace(0,1,6))
            cells[i, j].set_xlim(0.0, 1.0)
            cells[i, j].set_ylim(0.0, 1.0)
            cells[i, j].set_title("{}, AP: {:.3f}".format(classes[n*i+j+1], average_precisions[n*i+j+1]), fontsize=16)
    
    if not os.path.isdir("evaluate_result"):
        os.makedirs("evaluate_result")

    plt.savefig('evaluate_result/ssd300_face_detection.png')

if __name__ == "__main__":
    main()
