from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import time

from models.keras_ssd300 import ssd_300
from models.keras_mobileNet_ssd_ReLU6 import mobilenet_ssd_300
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

img_height = 300
img_width = 300

confidence_threshold = 0.7

K.clear_session() # Clear previous models from memory.

model = ssd_300(  image_size=(img_height, img_width, 3),
                            n_classes=20,
                            mode='inference',
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
                            confidence_thresh=confidence_threshold,
                            iou_threshold=0.45,
                            top_k=200,
                            nms_max_output_size=400)
                            
# 2: Load the trained weights into the model.

# TODO: Set the path of the trained weights.
weights_path = 'ssd300_train_models/ssd300_pascal_07+12_epoch-116_loss-4.5341_val_loss-4.2570.h5'
# weights_path = '/src/VGG_VOC0712Plus_SSD_300x300_iter_240000.h5'

model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

model_path = 'mn_ssd300_train_models_fd1/mn_ssd300_face_detection_epoch-118_loss-4.4522_val_loss-4.1524.h5'
model.load_weights(model_path, by_name=True)

WIDER_test_images_dir = 'dataset/WIDER_test/images/'
WIDER_test_filename = 'dataset/wider_face_split/wider_face_test_filelist.txt'

times = []

test_filenames = []

with open(WIDER_test_filename) as f:
    test_filenames = [line.rstrip() for line in f]

for _ in range(10):

    for num in range(6):
        input_images = [] # Store resized versions of the images here.
        row_images = []
        image_paths = []

        random_index = num

        row_images.append(imread(WIDER_test_images_dir + test_filenames[random_index]))
        image_paths.append(WIDER_test_images_dir + test_filenames[random_index])

        img = image.load_img(image_paths[0], target_size=(img_height, img_width))
        img = image.img_to_array(img)
        input_images.append(img)
        input_images = np.array(input_images)

        s_time = time.perf_counter()

        y_pred = model.predict(input_images)

        e_time = time.perf_counter()
        d_time = e_time - s_time
        d_time *= 1000

        times.append(d_time)

times = np.asarray(times)

time_mean = np.mean(times)

print("{} ms".format(time_mean))
