# -*- coding: utf-8 -*-

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from recognition import data_utils as du
from recognition import csv_utils as cu


def clear_session():
    K.clear_session()
    
    
def get_channels(color_mode):
    if color_mode == 'rgb':
        return 3
    else:
        return 1
    
    
def print_model_info(model_info):
    if float(model_info['rescale']) == 1./255:
        model_info['rescale'] = '1./255'
    print(model_info)
    
    
def freeze_layers(model, freeze_range):
    """
    Inputs:
      - model:
      - index: the index of the last freezed layer
    Return:
      - model: 
    """
    for index in freeze_range:
        model.layers[index].trainable = False
    return model


def load_my_model(model_path, return_model=True, return_info=True):
    model, model_info = None, None
    if return_model:
        model = load_model(model_path + '.h5')
    if return_info:
        model_info = cu.get_model_info(model_path + '.csv')
    return model, model_info


def my_generator(dataset_path, **kwargs):
    """Get the generator by the path of the dataset
    
    for more details or if you want to add some parameters, 
    please refer to 'https://keras.io/preprocessing/image/' 
    """
    rescale = kwargs.get('rescale', None)
    horizontal_flip = kwargs.get('horizontal_flip', False)
    vertical_flip = kwargs.get('vertical_flip', False)
    batch_size = kwargs.get('batch_size', 32)
    shuffle = kwargs.get('shuffle', True)
    target_size = kwargs.get('target_size', (224, 224))
    color_mode = kwargs.get('color_mode', 'rgb')
    class_mode = kwargs.get('class_mode', 'categorical')
    zoom_range = kwargs.get('zoom_range', 0.0)
    shear_range = kwargs.get('shear_range', 0.0)
    fill_mode = kwargs.get('fill_mode', 'nearest')
    rotation_range = kwargs.get('rotation_range', 0.0)
    width_shift_range = kwargs.get('width_shift_range', 0.0)
    height_shift_range = kwargs.get('height_shift_range', 0.0)
    channel_shift_range = kwargs.get('channel_shift_range', 0.0)
    
    data_gen = ImageDataGenerator(rescale=rescale,
                        horizontal_flip=horizontal_flip,
                        vertical_flip=vertical_flip,
                        shear_range=shear_range,
                        zoom_range=zoom_range,
                        rotation_range=rotation_range,
                        width_shift_range=width_shift_range,
                        height_shift_range=height_shift_range,
                        channel_shift_range=channel_shift_range,
                        fill_mode=fill_mode)
    
    generator = data_gen.flow_from_directory(dataset_path,
                                batch_size=batch_size, 
                                shuffle=shuffle, 
                                target_size=target_size, 
                                color_mode=color_mode, 
                                class_mode=class_mode)
    return data_gen, generator


def draw_plot(history, savefig_path=None):
    """ Draw the history curve
    
    Inputs:
      - history: A dictionary contains the history information
      - savefig_path: Decide whether to save the pictures
    """
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    x = np.arange(1, len(loss) + 1, 1)
    batch_size = history.history['batch_size']
    
    plt.figure(figsize=(10, 6))
    plt.title('{}, batch size = {}'.format('loss', batch_size))
    plt.plot(x, loss, 'g-', label='loss')
    plt.plot(x, val_loss, 'y-', label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    if savefig_path is not None:
        plt.savefig(savefig_path + 'loss.png', format='png', dpi=300)

    plt.figure(figsize=(10, 6))
    plt.title('{}, batch_size={}'.format('accuracy', batch_size))
    plt.plot(x, acc, 'b-', label='acc')
    plt.plot(x, val_acc, 'y-', label='val_acc')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.show()
    if savefig_path is not None:
        plt.savefig(savefig_path + 'accuracy.png', format='png', dpi=300)
        

def count_parameters(model):
    """ refer to the keras's function 'print_summary()'
    """
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))