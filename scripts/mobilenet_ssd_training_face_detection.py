from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import os
import pickle

from models.keras_mobileNet_ssd import mobilenet_ssd_300
# from models.keras_mobileNet_ssd300 import mobilenet_ssd_300
from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras.callbacks import TensorBoard

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
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

K.clear_session()

model, predictor_sizes = mobilenet_ssd_300(image_size=(img_height, img_width, img_channels),
                                       n_classes=n_classes,
                                       mode='training',
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
                                       swap_channels=swap_channels,
                                       return_predictor_sizes=True)

# weights_path = 'VGG_VOC0712Plus_SSD_512x512_iter_240000.h5'
weights_path = 'MobileNetSSD300weights_voc_2007.hdf5'
# weights_path = 'mn_ssd300_train_models/mn_ssd300_pascal_07+12_epoch-119_loss-3.2959_val_loss-2.9481.h5'

model.load_weights(weights_path, by_name=True)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

#####

# Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.

train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

WIDER_train_images_dir = 'dataset/WIDER_train/images/'
WIDER_val_images_dir = 'dataset/WIDER_val/images/'

WIDER_train_annotations_dir = 'dataset/WIDER_train/annotations/'
WIDER_val_annotations_dir = 'dataset/WIDER_val/annotations/'

WIDER_train_filename = 'dataset/wider_face_split/wider_face_train_filelist_small.txt'
WIDER_val_filename = 'dataset/wider_face_split/wider_face_val_filelist_small.txt'

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'face', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

if os.path.isfile('train_dataset.pickle'):
    with open('train_dataset.pickle', mode='rb') as f:
        train_dataset = pickle.load(f)

else:
    train_dataset.parse_xml(images_dirs=[WIDER_train_images_dir],
                            image_set_filenames=[WIDER_train_filename],
                            annotations_dirs=[WIDER_train_annotations_dir],
                            classes=classes,
                            include_classes='all',
                            exclude_truncated=False,
                            exclude_difficult=False,
                            ret=False)

    with open('train_dataset.pickle', mode='wb') as f:
        pickle.dump(train_dataset, f)

if os.path.isfile('val_dataset.pickle'):
    with open('val_dataset.pickle', mode='rb') as f:
        val_dataset = pickle.load(f)

else:
    val_dataset.parse_xml(images_dirs=[WIDER_val_images_dir],
                            image_set_filenames=[WIDER_val_filename],
                            annotations_dirs=[WIDER_val_annotations_dir],
                            classes=classes,
                            include_classes='all',
                            exclude_truncated=False,
                            exclude_difficult=True,
                            ret=False)

    with open('val_dataset.pickle', mode='wb') as f:
        pickle.dump(val_dataset, f)

batch_size = 8 # Change the batch size if you like, or if you run into GPU memory issues.

# For the training generator:
ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width,
                                            background=mean_color)

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(img_height, img_width)

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels,
                                                      resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

# Define model callbacks.
if not os.path.isdir("mn_ssd300_train_models_fd1"):
    os.makedirs("mn_ssd300_train_models_fd1")

model_checkpoint = ModelCheckpoint(filepath='/src/mn_ssd300_train_models_fd1/mn_ssd300_face_detection_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)

csv_logger = CSVLogger(filename='/src/mn_ssd300_train_models_fd1/mn_ssd300_face_detection_training_log.csv',
                       separator=',',
                       append=True)

# Define a learning rate schedule.
def lr_schedule(epoch):
    if epoch < 80:
        return 0.01
    elif epoch < 100:
        return 0.001
    else:
        return 0.0001

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

terminate_on_nan = TerminateOnNaN()

tb_cb = TensorBoard(log_dir="LOG/log8")

callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan,
             tb_cb  
            ]

# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch   = 0
final_epoch     = 120
# steps_per_epoch = 1000

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=ceil(train_dataset_size/batch_size),
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)