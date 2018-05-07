# -*- coding: utf-8 -*-

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, LambdaCallback
from keras.layers import BatchNormalization, Activation, Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU, PReLU
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.regularizers import l2
import csv
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd


def clear_session():
    """ Clear the seesion
    """
    K.clear_session()

    
def my_generator(path, **kwargs):
    """ Get the data generator by the path of the datset
    
    for more details or if you want to add some parameters, 
    please refer to 'https://keras.io/preprocessing/image/' 
    """
    rescale = kwargs.get('rescale', None)
    horizontal_flip = kwargs.get('horizontal_flip', False)
    batch_size = kwargs.get('batch_size', 32)
    shuffle = kwargs.get('shuffle', True)
    target_size = kwargs.get('target_size', (224, 224))
    color_mode = kwargs.get('color_mode', 'rgb')
    class_mode = kwargs.get('class_mode', 'categorical')
    zoom_range = kwargs.get('zoom_range', 0.0)
    shear_range = kwargs.get('shear_range', 0.0)
    fill_mode = kwargs.get('fill_mode', 'constant')
    rotation_range = kwargs.get('rotation_range', 0.0)
    
    gen = ImageDataGenerator(rescale=rescale,
                     horizontal_flip=horizontal_flip,
                     shear_range=shear_range,
                     zoom_range=zoom_range,
                     rotation_range=rotation_range,
                     fill_mode=fill_mode)
    
    generator = gen.flow_from_directory(path, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            target_size=target_size, 
                            color_mode=color_mode, 
                            class_mode=class_mode)
    return generator


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
    learning_rate = history.history['learning_rate']
    decay = history.history['decay']
    batch_size = history.history['batch_size']
    
    plt.figure(figsize=(10, 6))
    plt.title('{}, learning rate={}, lr_decay={}, batch size = {}'.format('loss', 
                                                   learning_rate, 
                                                   decay, 
                                                   batch_size))
    plt.plot(x, loss, 'g-', label='loss')
    plt.plot(x, val_loss, 'y-', label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    if savefig_path is not None:
        plt.savefig(savefig_path + 'loss.png', format='png', dpi=300)

    plt.figure(figsize=(10, 6))
    plt.title('{}, learninng rate={}, lr_decay={}, batch_size={}'.format('accuracy', 
                                                  learning_rate, 
                                                  decay,
                                                  batch_size))
    plt.plot(x, acc, 'b-', label='acc')
    plt.plot(x, val_acc, 'y-', label='val_acc')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.show()
    if savefig_path is not None:
        plt.savefig(savefig_path + 'accuracy.png', format='png', dpi=300)
    
    
def get_number_of_batches(samples, batch_size):
    """ Calculate the number of batches
    
    Inputs:
     - samples: The number of samples
     - batch_size: Batch size
     
    Return:
     - The number of batches
    """
    if samples % batch_size == 0:
        return samples // batch_size
    else:
        return samples // batch_size + 1


def train_model(model, train_generator, valid_generator, model_path, **kwargs):
    """ Train and save the best model
    
    Inputs:
      - model:
      - train_generator: The generator of the training set
      - valid_generator: The generator of the validation set
      - model_path:
      - kwargs:
          - batch_size:
          - learning_rate:
          - decay:
          - epochs:
          
    Return:
      - history: A dictionary contains the training phase information
    """
    lr_print_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: print(
        ' - lr: {}'.format(K.eval(model.optimizer.lr))))

    mdcheck = ModelCheckpoint(filepath=model_path, 
                     monitor='val_acc', save_best_only=True)
    callbacks = [mdcheck]

    batch_size = kwargs.get('batch_size', 32)
    learning_rate = kwargs.get('learning_rate', 0.01)
    decay = kwargs.get('decay', 1e-3)
    epochs = kwargs.get('epochs', 40)
    
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, 
            decay=decay, amsgrad=False)
    model.compile(loss='categorical_crossentropy', 
                  optimizer=adam, 
                  metrics=['accuracy'])

    train_steps = len(train_generator)
    valid_steps = len(valid_generator)
    history = model.fit_generator(train_generator, 
                        steps_per_epoch=train_steps,
                        epochs=epochs, 
                        validation_data=valid_generator,
                        validation_steps=valid_steps,
                        callbacks=callbacks)

    history.history['learning_rate'] = learning_rate
    history.history['decay'] = decay
    history.history['batch_size'] = batch_size
    
    return history


def freeze_layers(model, index=-1):
    """ Freeze some layers of the model
    Inputs:
      - model:
      - index: the index of the last freezed layer
    Return:
      - model: 
    """
    for layer in model.layers[:index+1]:
        layer.trainable = False
    for layer in model.layers[index+1:]:
        layer.trainable = True
    return model


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

    
def get_channels(color_mode):
    if color_mode == 'rgb':
        return 3
    else:
        return 1
    

def get_prediction(model, generator):
    """ Get a model's prediction for the dataset
    
    Inputs:
      - model: A pre-trained model
      - generator: A data generator of the dataset
      
    Return:
      - prediction: A numpy array, contains the prediction of the current 
              dataset by pre-trained model
    """
    steps = len(generator)
    prediction = model.predict_generator(generator, steps=steps, verbose=0)
    return prediction
    

def get_topk_indices(generator, prediction, k=5):
    """ Get the indices of top-k probability values 
    
    Inputs:
      - generator: A data generator of the dataset
      - prediction: The prediction of the dataset
      - k: 
    
    Return:
      - topk_indices: A list contains the top-k indices
    """
    samples = len(generator.classes)
    if k > samples:
        raise ValueError('k is greater than the number of samples.')
    if k <= 0:
        raise ValueError('k can\'t less than 0!')
    topk_indices = np.argsort(prediction, axis=1)[:, -k:]
    topk_indices = list(topk_indices)
    return topk_indices


def evaluate_topk_accuracy(generator, prediction=None, 
                  topk_indices=None, k=5):
    """ Evaluate the top-k accuracy
    
    Inputs:
      - generator: A data generator of the dataset
      - prediction: The prediction of the dataset
      - topk_indices: A list contains the top-k indices
      - k: 
      
    Returns:
      - correct_number: The Number of correctly identified
      - total_number: Total number of samples
    """
    if topk_indices is None and prediction is None:
        raise ValueError('prediction and topk_indices can\'t both be \'None\'')
    
    if topk_indices is None:
        topk_indices = get_topk_indices(generator, prediction, k=k)
        
    correct_number = 0
    total_number = 0
    for (indices, correct_class) in zip(topk_indices, generator.classes):
        if correct_class in indices:
            correct_number += 1
        total_number += 1
    print('Top-{} accuracy: {:.2%}'.format(k, correct_number / total_number))
    return correct_number, total_number


def get_filename_from_path(path):
    """ Get the filename from the path of the file.
    
    Input:
     - path:
     
    Return:
     - filename:
    """
    filename = path[path.rfind('\\')+1:]
    return filename
    
    
def get_topk_classes(generator, class_indices, prediction=None, 
              topk_indices=None, k=5):
    """ Get the classes of top-k probability values 
    
    Inputs:
      - generator: A data generator of the dataset
      - class_indices: A dictionary contains the mapping from 
            class names to class indices
      - prediction: A numpy array contains the prediction of the 
            dataset with shape(N, D), where N is the number of 
            samples and D is the number of classes
      - topk_indices: A list contains the top-k indices
      - k: 
    
    Return:
      - topk_classes: A list contains the top-k classes
    """
    if topk_indices is None and prediction is None:
        raise ValueError('prediction and topk_indices can\'t both be \'None\'')
    
    if topk_indices is None:
        topk_indices = get_topk_indices(generator, prediction, k=k)
        
    topk_classes = []
    for indices in topk_indices:
        classes = []
        for index in indices:
            classes.append(list(class_indices)[index])
        topk_classes.append(classes)
    return topk_classes


def predictions_ensemble(generators, predictions, algorithm='vote', k=5):
    """ Get the best top-k indices from the predictions.  
    
    Inputs:
     - generators: A list contains all pre-trained models' data generator
     - predictions: A list contains the predictions of all pre-trained models
     - algorithm: If algorithm is 'vote', adopt voting mechanism, otherwise 
             probability accumulation
     - k:
    
    Return:
     - topk_indices_ensemble: A list contains the top-k indices obtained 
             from the predictions
    """
    if algorithm not in {'vote', 'add'}:
        raise ValueError('The algorithm must be \'vote\' or \'add\'')
    
    if algorithm == 'vote':
        all_topk_indices = []
        for generator, prediction in zip(generators, predictions):
            all_topk_indices.append(get_topk_indices(generator, prediction, k=k))

        all_topk_indices = np.array(all_topk_indices)
        topk_indices_ensemble = []
        samples = len(generators[0].classes)
        for index in range(samples):
            (values, counts) = np.unique(all_topk_indices[:, index, :], 
                               return_counts=True)
            top_k = np.argsort(counts)[-k:]
            topk_indices_ensemble.append(values[top_k])
    else:
        predictions_sum = np.mean(predictions, axis=0)
        topk_indices_ensemble = get_topk_indices(generators[0], predictions_sum, k=k)
        
    return topk_indices_ensemble
        

def models_ensemble(models_list, data_path):
    """
    Inputs:
      - models_list: A list contains all pre-trained models' information
            Info: A tuple, (model_path, batch_size, (image_size, image_size), color_mode)
             
      - data_path: The path of the dataset
   
    Returns:
      - generators: A list contains generators for all pre-trained models 
      - predictions: A list contains predictions for all pre-trained models.
    """
    predictions, generators = [], []
    for model_params in models_list:
        (model_path, batch_size, target_size, color_mode) = model_params
        generator = my_generator(data_path, 
                         rescale=1./255, 
                         shuffle=False,
                         batch_size=batch_size, 
                         target_size=target_size, 
                         color_mode=color_mode)

        model = load_model(model_path)
        generators.append(generator)
        predictions.append(get_prediction(model, generator))
        
    return generators, predictions


def get_csv_format_data(test_generator, class_indices, topk_indices=None, 
                topk_classes=None):
    """ Get data in a csv format
    
    Inputs:
      - test_generator: The data generator of the test set
      - class_indices: A dictionary contains the mapping from class 
            names to class indices
      - topk_indices: A list contains the top-k indices
      
    Return:
      - csv_data: A list, 
    """
    if topk_indices is None and topk_classes is None:
        raise ValueError('topk_indices and topk_classes can\'t both be \'None\'')
    csv_data = []
    csv_data.append(['filename', 'label'])
    if topk_indices is not None:
        for path, indices in zip(test_generator.filenames, topk_indices):
            classes = []
            for index in indices:
                classes.append(list(class_indices.keys())[index])
            classes.reverse()
            # classes = classes[::-1]
            filename = get_filename_from_path(path)
            csv_data.append([filename, ''.join(classes)])
    
    elif topk_classes is not None:
        for path, classes in zip(test_generator.filenames, topk_classes):
            classes.reverse()
            # classes = classes[::-1]
            filename = get_filename_from_path(path)
            csv_data.append([filename, ''.join(classes)])
    
    return csv_data
    
    
def write_into_csv(data, csv_path=None):
    """Write the answer into a csv file
    
    Inputs:
      - data:
      - csv_path:
    """
    if type(data) not in {list, dict}:
        raise ValueError('The type of the input data must be \'dict\' or \'list\'')
        
    if csv_path is not None:
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            if type(data) is dict:
                for key, val in data.items():
                    w.writerow([key, val])
            else:
                w.writerows(data)
                
        print('Done!')
    else:
        raise ValueError('csv_path should not be \'None\'')
        
        
def get_class_indices(filepath):
    """
    Input:
     - filepath:
    Return:
     - class_indices: A dictionary, mapping from class names to class indices
    """
    data_frame = pd.read_csv(filepath, sep=',',header=None)
    class_indices = dict(data_frame.values)
    return class_indices