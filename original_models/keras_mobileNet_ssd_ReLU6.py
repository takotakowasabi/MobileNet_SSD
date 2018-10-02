from __future__ import division
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Activation, ReLU, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate, BatchNormalization, SeparableConv2D
from keras.layers.merge import concatenate
from keras.regularizers import l2
import keras.backend as K
from keras.applications import MobileNet

from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

def mobilenet_ssd_300(image_size,
            n_classes,
            mode='training',
            l2_regularization=0.0005,
            min_scale=None,
            max_scale=None,
            scales=None,
            aspect_ratios_global=None,
            aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]],
            two_boxes_for_ar1=True,
            steps=[8, 16, 32, 64, 100, 300],
            offsets=None,
            clip_boxes=False,
            variances=[0.1, 0.1, 0.2, 0.2],
            coords='centroids',
            normalize_coords=True,
            subtract_mean=[123, 117, 104],
            divide_by_stddev=None,
            swap_channels=[2, 1, 0],
            confidence_thresh=0.01,
            iou_threshold=0.45,
            top_k=200,
            nms_max_output_size=400,
            return_predictor_sizes=False):

    n_predictor_layers = 6 # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_classes += 1 # Account for the background class.
    l2_reg = l2_regularization # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1 & len(n_boxes) != 0:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    # print("Boxes:{}".format(n_boxes))

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)

    ############################################################################
    # Build the network.
    ############################################################################

    x = Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(x1)

    mobilenet=MobileNet(input_shape=(224,224,3),include_top=False,weights='imagenet')

    FeatureExtractor = Model(inputs=mobilenet.input, outputs=mobilenet.get_layer('conv_pw_5_relu').output)
    mobilenet_conv_pw_5_relu  = FeatureExtractor(x1)

    conv6dw = SeparableConv2D(256, (3, 3), padding='same', strides=(2, 2), name='conv_dw_6')(mobilenet_conv_pw_5_relu)
    conv6dw = BatchNormalization(momentum=0.99, name='conv_dw_6_bn')(conv6dw)
    conv6dw = ReLU(6., name='conv_dw_6_relu')(conv6dw)

    conv6pw = Conv2D(256, (1, 1), padding='same', name='conv_pw_6')(conv6dw)
    conv6pw = BatchNormalization(momentum=0.99, name='conv_pw_6_bn')(conv6pw)
    conv6pw = ReLU(6., name='conv_pw_6_relu')(conv6pw)

    conv7dw = SeparableConv2D(512, (3, 3), padding='same', name='conv_dw_7')(conv6pw)
    conv7dw = BatchNormalization(momentum=0.99, name='conv_dw_7_bn')(conv7dw)
    conv7dw = ReLU(6., name='conv_dw_7_relu')(conv7dw)

    conv7pw = Conv2D(512, (1, 1), padding='same', name='conv_pw_7')(conv7dw)
    conv7pw = BatchNormalization(momentum=0.99, name='conv_pw_7_bn')(conv7pw)
    conv7pw = ReLU(6., name='conv_pw_7_relu')(conv7pw)

    conv8dw = SeparableConv2D(512, (3, 3), padding='same', name='conv_dw_8')(conv7pw)
    conv8dw = BatchNormalization(momentum=0.99, name='conv_dw_8_bn')(conv8dw)
    conv8dw = ReLU(6., name='conv_dw_8_relu')(conv8dw)

    conv8pw = Conv2D(512, (1, 1), padding='same', name='conv_pw_8')(conv8dw)
    conv8pw = BatchNormalization(momentum=0.99, name='conv_pw_8_bn')(conv8pw)
    conv8pw = ReLU(6., name='conv_pw_8_relu')(conv8pw)

    conv9dw = SeparableConv2D(512, (3, 3), padding='same', name='conv_dw_9')(conv8pw)
    conv9dw = BatchNormalization(momentum=0.99, name='conv_dw_9_bn')(conv9dw)
    conv9dw = ReLU(6., name='conv_dw_9_relu')(conv9dw)

    conv9pw = Conv2D(512, (1, 1), padding='same', name='conv_pw_9')(conv9dw)
    conv9pw = BatchNormalization(momentum=0.99, name='conv_pw_9_bn')(conv9pw)
    conv9pw = ReLU(6., name='conv_pw_9_relu')(conv9pw)

    conv10dw = SeparableConv2D(512, (3, 3), padding='same', name='conv_dw_10')(conv9pw)
    conv10dw = BatchNormalization(momentum=0.99, name='conv_dw_10_bn')(conv10dw)
    conv10dw = ReLU(6., name='conv_dw_10_relu')(conv10dw)

    conv10pw = Conv2D(512, (1, 1), padding='same', name='conv_pw_10')(conv10dw)
    conv10pw = BatchNormalization(momentum=0.99, name='conv_pw_10_bn')(conv10pw)
    conv10pw = ReLU(6., name='conv_pw_10_relu')(conv10pw)

    conv11dw = SeparableConv2D(512, (3, 3), padding='same', name='conv_dw_11')(conv10pw)
    conv11dw = BatchNormalization(momentum=0.99, name='conv_dw_11_bn')(conv11dw)
    conv11dw = ReLU(6., name='conv_dw_11_relu')(conv11dw)

    conv11pw = Conv2D(512, (1, 1), padding='same', name='conv_pw_11')(conv11dw)
    conv11pw = BatchNormalization(momentum=0.99, name='conv_pw_11_bn')(conv11pw)
    conv11pw = ReLU(6., name='conv_pw_11_relu')(conv11pw)

    conv12dw = SeparableConv2D(512, (3, 3), strides=(2, 2), padding='same', name='conv_dw_12')(conv11pw)
    conv12dw = BatchNormalization(momentum=0.99, name='conv_dw_12_bn')(conv12dw)
    conv12dw = ReLU(6., name='conv_dw_12_relu')(conv12dw)

    conv12pw = Conv2D(1024, (1, 1), padding='same', name='conv_pw_12')(conv12dw)
    conv12pw = BatchNormalization(momentum=0.99, name='conv_pw_12_bn')(conv12pw)
    conv12pw = ReLU(6., name='conv_pw_12_relu')(conv12pw)

    conv13dw = SeparableConv2D(1024, (3, 3), padding='same', name='conv_dw_13')(conv12pw)
    conv13dw = BatchNormalization(momentum=0.99, name='conv_dw_13_bn')(conv13dw)
    conv13dw = ReLU(6., name='conv_dw_13_relu')(conv13dw)

    conv13pw = Conv2D(1024, (1, 1), padding='same', name='conv_pw_13')(conv13dw)
    conv13pw = BatchNormalization(momentum=0.99, name='conv_pw_13_bn')(conv13pw)
    conv13pw = ReLU(6., name='conv_pw_13_relu')(conv13pw)

    conv14_1 = Conv2D(256, (1, 1),  padding='same', name='conv14_1')(conv13pw)
    conv14_1 = BatchNormalization(momentum=0.99, name='bn14_1')(conv14_1)
    conv14_1 = ReLU(6., name='conv14_1_relu')(conv14_1)

    conv14_2dw = SeparableConv2D(512, (3, 3), strides=(2, 2), padding='same', name='conv_dw_14_2')(conv14_1)
    conv14_2dw = BatchNormalization(momentum=0.99, name='conv_dw_14_2_bn')(conv14_2dw)
    conv14_2dw = ReLU(6., name='conv_dw_14_2_relu')(conv14_2dw)

    conv14_2pw = Conv2D(512, (1, 1), padding='same', name='conv_pw_14_2')(conv14_2dw)
    conv14_2pw = BatchNormalization(momentum=0.99, name='conv_pw_14_2_bn')(conv14_2pw)
    conv14_2pw = ReLU(6., name='conv_pw_14_2_relu')(conv14_2pw)

    conv15_1 = Conv2D(128, (1, 1),  padding='same', name='conv15_1')(conv14_2pw)
    conv15_1 = BatchNormalization(momentum=0.99, name='bn15_1')(conv15_1)
    conv15_1 = ReLU(6., name='conv15_1_relu')(conv15_1)

    conv15_2dw = SeparableConv2D(256, (3, 3), name='conv_dw_15_2')(conv15_1)
    conv15_2dw = BatchNormalization(momentum=0.99, name='conv_dw_15_2_bn')(conv15_2dw)
    conv15_2dw = ReLU(6., name='conv_dw_15_2_relu')(conv15_2dw)

    conv15_2pw = Conv2D(256, (1, 1), padding='same', name='conv_pw_15_2')(conv15_2dw)
    conv15_2pw = BatchNormalization(momentum=0.99, name='conv_pw_15_2_bn')(conv15_2pw)
    conv15_2pw = ReLU(6., name='conv_pw_15_2_relu')(conv15_2pw)

    conv16_1 = Conv2D(128, (1, 1),  padding='same', name='conv16_1')(conv15_2pw)
    conv16_1 = BatchNormalization(momentum=0.99, name='bn16_1')(conv16_1)
    conv16_1 = ReLU(6., name='conv16_1_relu')(conv16_1)

    conv16_2 = SeparableConv2D(256, (3, 3), name='conv16_2_')(conv16_1)
    conv16_2 = BatchNormalization(momentum=0.99, name='bn16_2')(conv16_2)
    conv16_2 = ReLU(6., name='conv16_2_relu')(conv16_2)

    conv5_mbox_loc = Conv2D(n_boxes[0] * 4, (1,1), padding='same', name='conv5_mbox_loc')(mobilenet_conv_pw_5_relu)
    conv11_mbox_loc = Conv2D(n_boxes[1] * 4, (1,1), padding='same', name='conv11_mbox_loc_')(conv11pw)
    conv13_mbox_loc = Conv2D(n_boxes[2] * 4, (1,1), padding='same', name='conv13_mbox_loc_')(conv13pw)
    conv14_mbox_loc = Conv2D(n_boxes[3] * 4, (1,1), padding='same', name='conv14_mbox_loc_')(conv14_2pw)
    conv15_mbox_loc = Conv2D(n_boxes[4] * 4, (1,1), padding='same', name='conv15_mbox_loc_')(conv15_2pw)
    conv16_mbox_loc = Conv2D(n_boxes[5] * 4, (1,1), padding='same', name='conv16_mbox_loc_')(conv16_2)

    conv5_mbox_loc_reshape = Reshape((-1, 4), name='conv5_mbox_loc_reshape')(conv5_mbox_loc)
    conv11_mbox_loc_reshape = Reshape((-1, 4), name='conv11_mbox_loc_reshape')(conv11_mbox_loc)
    conv13_mbox_loc_reshape = Reshape((-1, 4), name='conv13_mbox_loc_reshape')(conv13_mbox_loc)
    conv14_mbox_loc_reshape = Reshape((-1, 4), name='conv14_mbox_loc_reshape')(conv14_mbox_loc)
    conv15_mbox_loc_reshape = Reshape((-1, 4), name='conv15_mbox_loc_reshape')(conv15_mbox_loc)
    conv16_mbox_loc_reshape = Reshape((-1, 4), name='conv16_mbox_loc_reshape')(conv16_mbox_loc)

    conv5_mbox_conf = Conv2D(n_boxes[0] * n_classes, (1,1), padding='same',name='conv5_mbox_conf')(mobilenet_conv_pw_5_relu)
    conv11_mbox_conf = Conv2D(n_boxes[1] * n_classes, (1,1), padding='same',name='conv11_mbox_conf_')(conv11pw)
    conv13_mbox_conf = Conv2D(n_boxes[2] * n_classes, (1,1), padding='same',name='conv13_mbox_conf_')(conv13pw)
    conv14_mbox_conf = Conv2D(n_boxes[3] * n_classes, (1,1), padding='same',name='conv14_mbox_conf_')(conv14_2pw)
    conv15_mbox_conf = Conv2D(n_boxes[4] * n_classes, (1,1), padding='same',name='conv15_mbox_conf_')(conv15_2pw)
    conv16_mbox_conf = Conv2D(n_boxes[5] * n_classes, (1,1), padding='same',name='conv16_mbox_conf_')(conv16_2)

    conv5_mbox_conf_reshape = Reshape((-1, n_classes), name='conv5_mbox_conf_reshape')(conv5_mbox_conf)
    conv11_mbox_conf_reshape = Reshape((-1, n_classes), name='conv11_mbox_conf_reshape')(conv11_mbox_conf)
    conv13_mbox_conf_reshape = Reshape((-1, n_classes), name='conv13_mbox_conf_reshape')(conv13_mbox_conf)
    conv14_mbox_conf_reshape = Reshape((-1, n_classes), name='conv14_mbox_conf_reshape')(conv14_mbox_conf)
    conv15_mbox_conf_reshape = Reshape((-1, n_classes), name='conv15_mbox_conf_reshape')(conv15_mbox_conf)
    conv16_mbox_conf_reshape = Reshape((-1, n_classes), name='conv16_mbox_conf_reshape')(conv16_mbox_conf)

    conv5_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                                       two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0], clip_boxes=clip_boxes,
                                       variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv5_mbox_priorbox')(mobilenet_conv_pw_5_relu)
    conv11_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                                       two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1], clip_boxes=clip_boxes,
                                       variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv11_mbox_priorbox')(conv11pw)
    conv13_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                                       two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2], clip_boxes=clip_boxes,
                                       variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv13_mbox_priorbox')(conv13pw)
    conv14_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                                       two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3], clip_boxes=clip_boxes,
                                       variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv14_mbox_priorbox')(conv14_2pw)
    conv15_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios[4],
                                       two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4], this_offsets=offsets[4], clip_boxes=clip_boxes,
                                       variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv15_mbox_priorbox')(conv15_2pw)
    conv16_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios[5],
                                       two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5], this_offsets=offsets[5], clip_boxes=clip_boxes,
                                       variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv16_mbox_priorbox')(conv16_2)

    conv5_mbox_priorbox_reshape = Reshape((-1, 8), name='conv5_mbox_priorbox_reshape')(conv5_mbox_priorbox)
    conv11_mbox_priorbox_reshape = Reshape((-1, 8), name='conv11_mbox_priorbox_reshape')(conv11_mbox_priorbox)
    conv13_mbox_priorbox_reshape = Reshape((-1, 8), name='conv13_mbox_priorbox_reshape')(conv13_mbox_priorbox)
    conv14_mbox_priorbox_reshape = Reshape((-1, 8), name='conv14_mbox_priorbox_reshape')(conv14_mbox_priorbox)
    conv15_mbox_priorbox_reshape = Reshape((-1, 8), name='conv15_mbox_priorbox_reshape')(conv15_mbox_priorbox)
    conv16_mbox_priorbox_reshape = Reshape((-1, 8), name='conv16_mbox_priorbox_reshape')(conv16_mbox_priorbox)

    mbox_loc = concatenate([conv5_mbox_loc_reshape,
                            conv11_mbox_loc_reshape,
                            conv13_mbox_loc_reshape,
                            conv14_mbox_loc_reshape,
                            conv15_mbox_loc_reshape,
                            conv16_mbox_loc_reshape], axis=1, name='mbox_loc')

    mbox_conf = concatenate([conv5_mbox_conf_reshape,
                             conv11_mbox_conf_reshape,
                             conv13_mbox_conf_reshape,
                             conv14_mbox_conf_reshape,
                             conv15_mbox_conf_reshape,
                             conv16_mbox_conf_reshape], axis=1, name='mbox_conf')

    mbox_priorbox = concatenate([conv5_mbox_priorbox_reshape, 
                                 conv11_mbox_priorbox_reshape, 
                                 conv13_mbox_priorbox_reshape, 
                                 conv14_mbox_priorbox_reshape, 
                                 conv15_mbox_priorbox_reshape, 
                                 conv16_mbox_priorbox_reshape], axis=1, name='mbox_priorbox')

    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    predictions = concatenate([mbox_conf_softmax, mbox_loc, mbox_priorbox], axis=2, name='predictions')

    if mode == 'training':
        model = Model(inputs=x, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    else:
        raise ValueError("`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    if return_predictor_sizes:
        predictor_sizes = np.array([conv5_mbox_conf._keras_shape[1:3],
                                    conv11_mbox_conf._keras_shape[1:3],
                                    conv13_mbox_conf._keras_shape[1:3],
                                    conv14_mbox_conf._keras_shape[1:3],
                                    conv15_mbox_conf._keras_shape[1:3],
                                    conv16_mbox_conf._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model