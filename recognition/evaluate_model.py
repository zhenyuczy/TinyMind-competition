# -*- coding: utf-8 -*-

import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from recognition import image_utils as iu 
from recognition import data_utils as du
from recognition import model_utils as mu

    
def predict_without_augmentation(model, generator):
    """Get the model's prediction for the dataset without 
        data augmentation.
    
    Inputs:
      - model: The pre-trained model.
      - generator: The data generator of the dataset.
      
    Return:
      - prediction: A numpy array, contains the prediction for 
              the current dataset by pre-trained model.
    """
    steps = len(generator)
    prediction = model.predict_generator(generator, 
                             steps=steps, 
                             verbose=0)
    return prediction
    
    
def get_generators_for_models_list(models_list, dataset_path):
    """Get each model's generator for the dataset.
    
    Input:
     - models_list: A list contains the pre_trained models.
     - dataset_path: A string, the path of the dataset.
     
    Return:
     - generators: A list contains all generators for the dataset.
    """
    gens, generators = [], []
    
    for model_name in models_list:
        _, model_info = mu.load_my_model(model_name, return_model=False)
        batch_size = int(model_info['batch_size'])
        rescale = float(model_info['rescale'])
        image_size = int(model_info['image_size'])
        color_mode = model_info['color_mode']
        algorithm = model_info['algorithm']
        
        print('Obtaining the generator of the model \'{}\', the model\'s information are as follows: '.format(model_name))
        mu.print_model_info(model_info)
        
        
        if algorithm == 'no_aug' or algorithm == 'my_aug':
            gen, generator = mu.my_generator(dataset_path,
                                  shuffle=False,
                                  rescale=rescale, 
                                  batch_size=batch_size, 
                                  target_size=(image_size, image_size), 
                                  color_mode=color_mode)
        elif algorithm == 'ke_aug':
            
            horizontal_flip = model_info['horizontal_flip']=='True'
            vertical_flip = model_info['vertical_flip']=='True'
            width_shift_range = float(model_info['width_shift_range'])
            height_shift_range = float(model_info['height_shift_range'])
            rotation_range = int(model_info['rotation_range'])
            shear_range = float(model_info['shear_range'])
            zoom_range = float(model_info['zoom_range'])
            channel_shift_range = float(model_info['channel_shift_range'])
            shuffle = model_info['shuffle']=='True'
                                
            gen, generator = mu.my_generator(dataset_path, 
                                  rescale=rescale,
                                  horizontal_flip=horizontal_flip,
                                  vertical_flip=vertical_flip,
                                  width_shift_range=width_shift_range,
                                  height_shift_range=height_shift_range,
                                  rotation_range=rotation_range,
                                  shear_range=shear_range,
                                  zoom_range=zoom_range,
                                  shuffle=shuffle,
                                  batch_size=batch_size, 
                                  color_mode=color_mode, 
                                  target_size=(image_size, image_size))
        
        gens.append(gen)
        generators.append(generator)
    print('All the generators have been obtained!')
    return gens, generators
    

def save_predictions_for_models_list(models_list, predictions, save_dir):
    """Save each model's prediction for the dataset. 

    Inputs:
     - models_list: A list contains all pre_trained models.
     - predictions: A list contains all pre_trained models' predictions.
     - save_dir: The dictionary used to store the predicitons.
    """
    for model_name, prediction in zip(models_list, predictions):
        index = model_name.rfind('/')
        left = model_name[:index]
        right = model_name[index+1:]
        prediction_name = left[left.rfind('/')+1:] + '_' + right
        prediction_path = save_dir + '/' + prediction_name + '.npy'
        print('Saving the prediction in the file \'{}\''.format(prediction_path))
        np.save(open(prediction_path, 'wb'), prediction)
    print('All the preditions have been saved!')
    

def get_predictions_from_backup_for_models_list(models_list, storage_dir):
    """Get predictions from the storage directory.
    
    Input:
     - models_list: A list contains all pre_trained models.
     - storage_dir: A string, the path of the storange directory.
     
    Return:
     - predicitons: A list contains all models' prediction.
    """
    predictions = []
    for model_name in models_list:
        index = model_name.rfind('/')
        left = model_name[:index]
        right = model_name[index+1:]
        prediction_name = left[left.rfind('/')+1:] + '_' + right
        prediction_path = storage_dir + '/' + prediction_name + '.npy'
        print('Loading prediction from the file \'{}\''.format(prediction_path))
        predictions.append(np.load(open(prediction_path, 'rb')))
    print('All the predictions have been Loaded!')
    return predictions

    
def get_topk_indices_by_single_generator(generator, prediction, k=5):
    """ Get the indices of top-k probability values.
    
    Inputs:
      - generator: A data generator of the dataset.
      - prediction: A numpy array, the prediction of the dataset.
      - k: An integer.
    
    Return:
      - topk_indices: A numpy array, the top-k indices.
    """
    samples = len(generator.classes)
    if k > samples:
        raise ValueError('k is greater than the number of samples.')
    if k <= 0:
        raise ValueError('k can\'t less than 0!')
    topk_indices = np.argsort(prediction, axis=1)[:, -k:]
    return topk_indices


def evaluate_topk_accuracy_by_single_generator(generator, 
                               class_indices,
                               prediction=None, 
                               topk_indices=None, 
                               k=5):
    """ Evaluate the top-k accuracy
    
    Inputs:
      - generator: The data generator of the dataset
      - class_indices: A dictionary contains the mapping from classes to indices.
      - prediction: The predictions of the dataset.
      - topk_indices: A numpy array contains the top-k indices.
      - k: 
      
    Returns:
      - correct_number: The number of correctly identified.
      - total_number: Total number of samples.
      - topk_indices: A numpy array contains the top-k indices.
      - topk_classes: A numpy array contains the top-k classes.
      - wrong_info: A list contains the wrong information.
    """
    if topk_indices is None and prediction is None:
        raise ValueError('prediction and topk_indices can\'t both be \'None\'')
        
    if topk_indices is None:
        topk_indices = get_topk_indices_by_single_generator(generator, 
                                           prediction, 
                                           k=k)
            
    correct_number, total_number, wrong_info = 0, 0, []
    topk_classes = get_topk_classes_by_single_generator(generator, 
                                       class_indices, 
                                       topk_indices=topk_indices,
                                       prediction=prediction, 
                                       k=k)
  
    for topk_class, filename in zip(topk_classes, generator.filenames):
        class_name = du.get_class_name_from_path(filename)
        file_name = du.get_file_name_from_path(filename)
        if class_name in topk_class:
            correct_number += 1
        else:
            wrong_info.append({'filename': file_name, 
                         'correct': class_name, 
                         'predict': list(topk_class)})
        total_number += 1
    
    print('Top-{} accuracy: {:.2%}'.format(k, correct_number / total_number))
    
    return total_number, correct_number, topk_indices, topk_classes, wrong_info
    
    
def get_topk_classes_by_single_generator(generator, class_indices, 
                           prediction=None, 
                           topk_indices=None, k=5):
    """ Get the classes of top-k probability values 
    
    Inputs:
      - generator: A data generator of the dataset
      - class_indices: A dictionary contains the mapping from 
            class names to class indices
      - prediction: A numpy array contains the prediction of the 
            dataset with shape(N, D), where N is the number of 
            samples and D is the number of classes
      - topk_indices: A numpy array contains the top-k indices.
      - k: An integer.
    
    Return:
      - topk_classes: A numpy array contains the top-k classes.
    """
    if topk_indices is None and prediction is None:
        raise ValueError('prediction and topk_indices can\'t both be \'None\'')
    
    if topk_indices is None:
        topk_indices = get_topk_indices_by_single_generator(generator, 
                                           prediction, 
                                           k=k)
    topk_classes = []
    for indices in topk_indices:
        classes = []
        for index in indices:
            # classes.append(list(class_indices)[index])
            classes.append(list(class_indices.keys())[index]) 
        topk_classes.append(classes)
    topk_classes = np.array(topk_classes)
    return topk_classes


def predict_by_keras_augmentation(model, 
                       dataset_path,
                       gen,
                       generator,
                       batch_process=32,
                       show_first_fig=False,
                       augment_size=6, 
                       num_classes=100,
                       color_mode='rgb',
                       target_size=(224, 224)):
    """ Predict by Keras's augmentation.
    Inputs:
      - model: The pre_trained model.
      - dataset_path: A string, path of the dataset.
      - gen: The method of the augmentation.
      - generator: A data generator.
      - batch_size: An integer, the size for each batch.
    
    Return:
      - prediction: A numpy array contains the model's prediction 
              by Keras's augmentation.
    """
    samples = len(generator.filenames)
    sample_index = 0
    batch_sample_index = 0
    augment_patches = []
    
    prediction = np.zeros((samples, num_classes))
    for index, file_name in enumerate(generator.filenames):
        file_path = du.join_path(dataset_path, file_name)
        if color_mode is 'rgb':
            img = image.load_img(file_path, 
                          grayscale=False, 
                          target_size=target_size)  
        else:
            img = image.load_img(file_path, 
                          grayscale=True, 
                          target_size=target_size)
        x = image.img_to_array(img)  
        x = np.expand_dims(x, axis=0)  
        
        sample_index += 1
        batch_sample_index += 1

        for number, augment_img in enumerate(gen.flow(x)):
            augment_patches.append(np.squeeze(augment_img))
            number += 1
            if number >= augment_size:
                break
        
        if show_first_fig is True:
            iu.show_augment_patches(augment_patches, color_mode=color_mode)
            break
            
        if sample_index % batch_process == 0 or sample_index == samples:
            augment_patches = np.array(augment_patches)
            if color_mode == 'grayscale':
                augment_patches = np.expand_dims(augment_patches, axis=3)
            augment_prediction = model.predict(augment_patches)
            augment_prediction = np.reshape(augment_prediction, 
                                  (batch_sample_index, augment_size, num_classes))
            prediction[sample_index-batch_sample_index:sample_index, :] = np.mean(
                augment_prediction, axis=1)
            
            augment_patches = []
            batch_sample_index = 0
            
    return prediction


def predict_by_my_augmentation(model,
                     dataset_path,
                     generator, 
                     batch_process=32,
                     num_classes=100,
                     show_first_fig=False,
                     rescale=1./255,
                     color_mode='rgb', 
                     target_size=(224, 224)):
    """ Predict by the data augmentation. This augmentation does not use Keras API.
    
    Inputs:
      - model:
      - dataset_path: A string, the path of the dataset.
      - generator: A generator of the dataset.
      - batch_size: An inetger, the maximum size of each batch.
      - show_first_fig: True or False, whether show the first augmented image.
      - rescale: The rescale parameter of the generator.
      - color_mode: 'rgb' or 'grayscale'.
      - target_size: A tuple(height, width).
      
    Return:
      - prediction: A numpy array, shape(samples, num_classes), contains 
              the predictions for dataset.
    """
    augment_patches = []
    augment_size = 7
    # weights = np.array([[[1./24], [1./24], [1./24], [1./24], [1./24], [1./24], [1./24]]])
    samples = len(generator.filenames)
    sample_index = 0
    batch_sample_index = 0
    
    prediction = np.zeros((samples, num_classes))
    
    for file in generator.filenames:
        filepath = du.join_path(dataset_path, file)

        if color_mode == 'rgb':
            image_data = iu.cv_imread(filepath, color_mode=color_mode)
            height, width, _ = image_data.shape
            image_data = iu.convert_RGB_to_BGR(image_data)
        else:
            image_data = iu.cv_imread(filepath, color_mode=color_mode)
            height, width = image_data.shape

        sample_index += 1
        batch_sample_index += 1
        
        # original
        # augment_patches.append(iu.resize_image_by_size(image_data, target_size))
        
        # 0.9 crop
        crop_height = int(height*0.9)
        crop_width = int(width*0.9)
        margin_height = (height-crop_height) // 2
        margin_width = (width-crop_width) // 2
        middle = image_data[margin_height:height-margin_height,
                     margin_width:width-margin_width]
        middle = iu.resize_image_by_size(middle, target_size)
        augment_patches.append(middle)
        
        left = image_data[:, :crop_width]
        left = iu.resize_image_by_size(left, target_size)
        augment_patches.append(left)

        right = image_data[:, width-crop_width:]
        right = iu.resize_image_by_size(right, target_size)
        augment_patches.append(right)
        
        top = image_data[:crop_height, :]
        top = iu.resize_image_by_size(top, target_size)
        augment_patches.append(top)
        
        bottom = image_data[height-crop_height:, :]
        bottom = iu.resize_image_by_size(bottom, target_size)
        augment_patches.append(bottom)
        
        # rotation 10
        rotate = iu.rotate_image_by_angle(image_data, 
                               need_resize=True, 
                               target_size=target_size, 
                               angle=10,
                               fill_color=(255, 255, 255))
        augment_patches.append(rotate)
        # rotation -10
        rotate = iu.rotate_image_by_angle(image_data, 
                               need_resize=True, 
                               target_size=target_size, 
                               angle=-10,
                               fill_color=(255, 255, 255))
        augment_patches.append(rotate)

        if show_first_fig is True:
            iu.show_augment_patches(augment_patches, color_mode=color_mode)
            break
            
        if sample_index % batch_process == 0 or sample_index == samples:
            augment_patches = np.array(augment_patches)
            if color_mode == 'grayscale':
                augment_patches = np.expand_dims(augment_patches, axis=3)
            augment_patches = augment_patches * rescale
            augment_prediction = model.predict(augment_patches)
            augment_prediction = np.reshape(augment_prediction, 
                                  (batch_sample_index, augment_size, num_classes))
            prediction[sample_index-batch_sample_index:sample_index, :] = np.mean(
                augment_prediction, axis=1)
            
            augment_patches = []
            batch_sample_index = 0
        
    return prediction


def predictions_ensemble(generators, predictions, algorithm='vote', k=5):
    """ Get the best top-k indices from the predictions.  
    
    Inputs:
     - generators: A list contains all pre-trained models' data generator.
     - predictions: A list contains the predictions of all pre-trained models.
     - algorithm: 'vote' or 'mean'.
     - k: An integer.
    
    Return:
     - ensemble_prediction:
     - ensemble_topk_indices: The top-k indices obtained from the predictions.
    """
    if algorithm not in {'vote', 'mean'}:
        raise ValueError('The algorithm must be \'vote\' or \'mean\'')
    
    ensemble_prediction = None
    
    if algorithm == 'vote':
        all_topk_indices = []
        for generator, prediction in zip(generators, predictions):
            all_topk_indices.append(get_topk_indices_by_single_generator(generator, 
                                                     prediction, 
                                                     k=k))

        all_topk_indices = np.array(all_topk_indices)
        ensemble_topk_indices = []
        samples = len(generators[0].classes)
        for index in range(samples):
            (values, counts) = np.unique(all_topk_indices[:, index, :], 
                               return_counts=True)
            top_k = np.argsort(counts)[-k:]
            ensemble_topk_indices.append(values[top_k])
    else:
        ensemble_prediction = np.mean(predictions, axis=0)
        ensemble_topk_indices = get_topk_indices_by_single_generator(generators[0], 
                                                 ensemble_prediction, 
                                                 k=k)
        
    return ensemble_prediction, ensemble_topk_indices
        

def get_generators_and_predictions_for_models_list(models_list, 
                                  dataset_path, 
                                  batch_process=32, 
                                  num_classes=100):
    """ Get each model's generator and prediction for the dataset in the models list.
    
    Inputs:
      - models_list: A list contains all pre-trained models' information.
      - dataset_path: The path of the dataset.
      - batch_process: 
      - num_classes: An integer, the number of classes.
   
    Returns:
      - generators: A list contains all pre-trained models' genegrators for 
              the dataset. 
      - predictions: A list contains all pre-trained models' predictions for 
              the dataset.
    """
    
    gens, generators = get_generators_for_models_list(models_list, dataset_path)
    predictions = []
    for model_name, gen, generator in zip(models_list, gens, generators): 
        model, model_info = mu.load_my_model(model_name)
       
        batch_size = int(model_info['batch_size'])
        rescale = float(model_info['rescale'])
        image_size = int(model_info['image_size'])
        color_mode = model_info['color_mode']
        algorithm = model_info['algorithm']
        
        augment_size = None
        if algorithm == 'ke_aug':
            augment_size = int(model_info['augment_size'])
            
        print('Using model \'{}\' to predict...'.format(model_name))
        
        prediction = get_prediction_by_pre_trained_model(model, 
                                         generator, 
                                         dataset_path=dataset_path, 
                                         gen=gen, 
                                         batch_process=batch_process, 
                                         num_classes=num_classes, 
                                         show_first_fig=False,
                                         color_mode=color_mode, 
                                         rescale=rescale, 
                                         augment_size=augment_size,
                                         target_size=(image_size, image_size), 
                                         algorithm=algorithm)
        predictions.append(prediction)
        del model
    print('Get all the generators and predictions for models\' parameters!')
    return generators, predictions


def get_prediction_by_pre_trained_model(model, generator, 
                           dataset_path=None, 
                           gen=None, 
                           batch_process=32, 
                           num_classes=100, 
                           show_first_fig=False,
                           color_mode='rgb', 
                           rescale=1./255, 
                           augment_size=6,
                           target_size=(224, 224), 
                           algorithm='no_aug'):
    """Get prediction by pre_trained model.
    
    Inputs:
      - model: The pre_trained model.
      - generator: A generator of the dataset.
      - dataset_path: A string, the path of the dataset.
      - gen: The augmentation of the generator.
      - batch_process: An inetger, the maximum size of each batch.
      - num_classes: An integer, number of classes.
      - show_first_fig: True or False, whether show the first augmented image, 
              only used when algorithm is 'my_aug' or 'ke_aug'.
      - augment_size: An integer, number of the test images by keras 
              augmentation. 
      - rescale: The rescale parameter of the generator.
      - color_mode: 'rgb' or 'grayscale'.
      - target_size: A tuple(height, width).
      - algorithm: 'no_aug', 'my_aug', 'ke_aug'.
              'no_aug': predict without augmentation.
              'my_aug': predict by my augmentation.
              'ke_aug': predict by Keras's augmentation.
    
    Return:
      - prediction: A numpy array contains the pre_trained model's prediction. 
      
    """
    
    if algorithm not in ['no_aug', 'my_aug', 'ke_aug']:
        raise ValueError('The algorithm must be one of \'no_aug\',\'my_aug\',\'ke_aug\'.')
        
    if algorithm == 'no_aug':
        prediction = predict_without_augmentation(model, generator)
        
    elif algorithm == 'my_aug':
        prediction = predict_by_my_augmentation(model,
                                   dataset_path,
                                   generator, 
                                   batch_process=batch_process,
                                   num_classes=num_classes,
                                   show_first_fig=show_first_fig,
                                   rescale=rescale,
                                   color_mode=color_mode, 
                                   target_size=target_size)
    else:
        prediction = predict_by_keras_augmentation(model, 
                                     dataset_path,
                                     gen,
                                     generator,
                                     batch_process=batch_process,
                                     show_first_fig=show_first_fig,
                                     augment_size=augment_size, 
                                     num_classes=num_classes,
                                     color_mode=color_mode,
                                     target_size=target_size)
        
    return prediction
