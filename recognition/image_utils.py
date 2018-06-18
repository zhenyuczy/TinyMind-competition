# -*- coding: utf-8 -*-


from recognition import data_utils as du
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.preprocessing import image


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def cv_imread(image_path, color_mode='rgb'):
    """ Uses opencv to read the image path which contains Chinese.
    
    Inputs:
      - image_path: A string, path of the image.
      - color_mode: 'rgb' or 'grayscale'.
      
    Return:
      - image_data: A numpy array, data of the image.
    """
    if color_mode not in {'rgb', 'grayscale'}:
        raise ValueError('color_mode must be \'rgb\' or \'grayscale\'.')
        
    if color_mode == 'rgb':
        image_data = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 
                          cv2.IMREAD_COLOR)
    else:
        image_data = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 
                          cv2.IMREAD_GRAYSCALE)
    return image_data


def convert_RGB_to_BGR(image_data):
    image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
    return image_data


def convert_BGR_to_RGB(image_data):
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    return image_data


def resize_image_by_size(image, target_size):
    resize_image = cv2.resize(image, target_size)
    return resize_image


def rotate_image_by_angle(image, 
                 need_resize=False, 
                 target_size=(224, 224), 
                 fill_color=(255, 255, 255),
                 angle=0):
    """Rotate image.
    
    Inputs: 
      - image: A numpy array contains image data with 
          shape(height, width, channels).
      - need_resize: True or False, whether resize the image.
      - target_size: A tuple (height, width) represents the size of 
          the output images.
      - fill_color: The color used to fill in missing pixels of the 
          rotated image.
      - angle: A float number between 0 and 360.
      
    Return:
      - rotate_image: A numpy array contains the rotated image data.
    """
    height, width = image.shape[0], image.shape[1]
    
    center = (height//2, width//2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotate_image = cv2.warpAffine(image, 
                        matrix, 
                        (width, height), 
                        borderValue=fill_color)
    if need_resize:
        rotate_image = resize_image_by_size(rotate_image, target_size)
    return rotate_image


def check_image(path, color_mode='rgb'):
    """ Check if the image is dirty.
    
    Inputs:
      - path: A string, the path of the image to check.
      - color_mode: 'rgb' or 'grayscale'.
      
    Return:
      - True or False, represents the check result.
    """
    img = cv_imread(path, color_mode=color_mode)
    r_sum = 0
    color_interval = []
    for i in range(15, -1, -1):
        number = np.sum(img >= i * 16) - r_sum
        color_interval.append(number)
        r_sum += number
    color_interval = np.array(color_interval)
    color_interval = np.sort(color_interval)
    if color_interval[14] / r_sum < 1./64:
        return False
    else:
        return True
    
    
def show_dirty_images(dirty_image_data, 
               visualizes_dirty_images=False, 
               print_dirty_images_info=False, 
               visualizes_number=40):
    """ Show the dirty images.
    
    Inputs:
      - dirty_data: A list, data of all dirty images.
      - visualizes_dirty_images: True or False, whether visualizes 
          dirty images.
      - print_dirty_images_info: True or False, whether prints the 
          information of dirty images.
      - visualizes_number: An integer, the number of image visualization.
    """
    total = len(dirty_image_data)
    if total > visualizes_number:
        total = visualizes_number
    
    if visualizes_dirty_images:
        col = 5
        if total % col == 0:
            row = total // col
        else:
            row = total // col + 1
        plt.figure(figsize=(20, 4 * row))
    
    if print_dirty_images_info:
        print('label' + '   ' + 'name')
        
    for index, (path, label) in enumerate(dirty_image_data[:total]):
        if visualizes_dirty_images:
            image_data = cv_imread(path)     
            plt.subplot(row, col, index + 1)
            plt.imshow(image_data)
            plt.title('label: {}'.format(label))
            plt.axis('off')
            
        if print_dirty_images_info:
            file_name = du.get_file_name_from_path(path)
            print(str(label) + '      ' + str(file_name))
    
    
def show_images_in_wrong_info(wrong_info, 
                    dataset_path, 
                    visualizes_number=40):
    """Show worng images.
    
    Inputs:
     - wrong_info: A dictionary contains the information of all 
         wrong images. 
     - dataset_path: A string contains the path of the dataset.
     - visualizes_number: An integer, the number of image visualization.
    """
    total = len(wrong_info)
    if total > visualizes_number:
        total = visualizes_number
        
    col = 5
    row = du.get_number_of_batches(total, col)
    
    plt.figure(figsize=(20, 4 * row))
    for index, info in enumerate(wrong_info[:total]):
        filename, correct, predict = info['filename'], info['correct'], info['predict']
        predict = ''.join(predict)
        filepath = du.join_path(du.join_path(dataset_path, correct), filename)
        image_data = cv_imread(filepath)
        plt.subplot(row, col, index + 1)
        plt.imshow(image_data)
        plt.title('correct:{},predict:{}'.format(correct, predict))
        plt.axis('off')
        
        
def show_augment_patches(augment_patches, color_mode='rgb'):
    """Show augmented data. This augmentation does not use the Keras API.
    
    Input:
     - augment_patches: A list contains the augmented data for single image.
    """
    patch_size = len(augment_patches)
    plt.figure(figsize=(4*patch_size, 4*patch_size))
    for index, patch in enumerate(augment_patches):
        plt.subplot(1, patch_size, index + 1)
        if color_mode == 'grayscale':
            plt.imshow(patch, cmap='gray')
        else:
            plt.imshow(patch)
        plt.axis('off')
        
        
def show_image_by_keras_augmentation(gen, 
                         file_path, 
                         target_size=(224, 224), 
                         visualizes_number=10):
    """Show augmented data. This augmentation uses Keras API.
    
    Inputs:
      - gen: The data augmentation method.
      - file_path: A string, the path of the image.
      - target_size: A tuple(height, width) represents the size of the 
          visualized image.
      - visualizes_number: An integer, the number of image visualization. 
    """
    img = image.load_img(file_path, target_size=target_size)
    x = image.img_to_array(img)  
    x = np.expand_dims(x, axis=0)  
    col = 5
    if visualizes_number % col == 0:
        row = visualizes_number // col
    else:
        row = visualizes_number // col + 1
    plt.figure(figsize=(4 * col, 3 * row))
    for count, augment_img in enumerate(gen.flow(x)):
        count += 1
        plt.subplot(row, col, count)
        plt.imshow(np.squeeze(augment_img))
        plt.axis('off')
        if count >= visualizes_number:
            break