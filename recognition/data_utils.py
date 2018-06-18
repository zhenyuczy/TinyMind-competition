# -*- coding: utf-8 -*-


from recognition import image_utils as iu
import numpy as np
import os
import shutil


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + \
                                np.abs(y))))

def join_path(path1, path2):
    """ Join paths  """
    path = os.path.join(path1, path2)
    return path


def get_number_of_batches(samples, batch_size):
    """ Get the number of batches.
    
    Inputs: 
     - smaples: An integer, the number of samples.
     - batch_size: An interger, the number of each batch.
     
    Return:
     - batch_num: An integer, the number of batches.
    """
    batch_num = 0
    if samples % batch_size == 0:
        batch_num = samples // batch_size
    else:
        batch_num = samples // batch_size + 1
    return batch_num
    

def get_file_name_from_path(path):
    """ Get the file name from the file path.
    
    Input:
     - path: A string contains the file path.
     
    Return:
     - file_name: A string contains the file name.
    """
    file_name = path[path.rfind('\\')+1:]
    return file_name
   
    
def get_class_name_from_path(path):
    """ Get the class name from the file path.
    
    Input:
     - path: A string contains the file path.
    Return:
     - class_name: A string contains the class name.
    """
    class_name = path[:path.rfind('\\')]
    return class_name
    
    
def split_dataset_into_training_and_test_sets(data_set_dir, 
                               training_set_dir, 
                               test_set_dir, 
                               test_set_size=0.1):
    
    """ Split dataset.
    
    Inputs:
      - data_set_dir: The directory of the source data set.
          the structure of the directory must be:
          directory/
                label1/
                    label1_001.jpg
                    label2_002.jpg
                    ...
                label2/
                    label2_001.jpg
                    label2_002.jpg
                    ...
                 ... 
      - training_set_dir: The directory of the training set.
      - test_set_dir: The directory of the test set.
      - test_set_size: A real number, the size of the test set.
      
    Returns:
      - training_set_files: The number of files in the training set.
      - test_set_files: The number of files in the test set.
    """
    if test_set_size<0 or test_set_size>1:
        raise ValueError('Test size must be a real number between 0 and 1.')
        
    if not os.path.exists(data_set_dir):
        raise ValueError('The data directory {} does not exist!'.format(data_set_dir))
        
    if os.path.exists(training_set_dir):
        shutil.rmtree(training_set_dir, ignore_errors=False)
    os.makedirs(training_set_dir)
    
    if os.path.exists(test_set_dir):
        shutil.rmtree(test_set_dir, ignore_errors=False) 
    os.makedirs(test_set_dir)
    
    training_set_files = 0
    test_set_files = 0
    
    all_classes = []
    training_set_data_statistics = []
    test_set_data_statistics = []
    
    for root, subdirs, files in os.walk(data_set_dir):
        class_name = os.path.basename(root)
        total_files = len(files)
        if total_files is 0:
            continue
        
        temp_training_class_dir = os.path.join(training_set_dir, class_name)
        temp_test_class_dir = os.path.join(test_set_dir, class_name)
                        
        if not os.path.exists(temp_training_class_dir):
            os.makedirs(temp_training_class_dir)
                
        if not os.path.exists(temp_test_class_dir):
            os.makedirs(temp_test_class_dir)

        mark = np.ones((total_files, ), dtype=int)
        test_number = round(total_files*test_set_size)
        mask = np.random.choice(total_files, test_number, replace=False)
        mark[mask] = 0

        training_files_for_temp_class = 0
        test_files_for_temp_class = 0
        for index, file in enumerate(files):
            file_path = os.path.join(root, file)
            if mark[index] == 1:
                shutil.copy(file_path, temp_training_class_dir + '/' + file)
                training_files_for_temp_class += 1
            else:
                shutil.copy(file_path, temp_test_class_dir + '/' + file)
                test_files_for_temp_class += 1
                
        training_set_files += training_files_for_temp_class
        test_set_files += test_files_for_temp_class
        all_classes.append(class_name)
        training_set_data_statistics.append(training_files_for_temp_class)
        test_set_data_statistics.append(test_files_for_temp_class)
        
    print('The number of files in the source data set: {}'.format(
        training_set_files + test_set_files))
    print('The number of files in the training set: {}'.format(training_set_files))
    print('The number of files in the test set: {}'.format(test_set_files))
    
    print('All the classes in the source data set:')
    print(all_classes)
    print('The number of files for each class in the training set:')
    print(training_set_data_statistics)
    print('The number of files for each class in the test set:')
    print(test_set_data_statistics)
    
    return training_set_files, test_set_files


def search(directory, paths_list, labels_list, label=None):
    """ search the directory and get the path and label of all image files
    
    Inputs:
      - directory: A folder, the structure of it must be:
        directory/
            label1/
                label1_001.jpg
                label2_002.jpg
                ...
            label2/
                label2_001.jpg
                label2_002.jpg
                ...
            ...
      - paths_list: A list contains the path of all image files.
      - labels_list: A list contains the label of all image files.
      - label: If label is None, represents that there are only folders in the 
              current directory. Otherwise it represents the label of all files 
              in the current directory.
    
    """
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isdir(file_path):
            search(file_path, paths_list, labels_list, label=file)
        elif file[-4:] in {'.png', '.jpg'}:
            paths_list.append(file_path)
            labels_list.append(label)
            
            
def get_paths_and_labels_from_directory(directory):
    """Obtains the path and label of all image files
    
    Input:
      - directory: A folder, the structure of it must be:
        directory/
            label1/
                label1_001.jpg
                label2_002.jpg
                ...
            label2/
                label2_001.jpg
                label2_002.jpg
                ...
            ...
   Returns: 
      - paths_list: A list contains the path of all image files.
      - labels_list: A list contains the label of all image files.
    """
    paths_list, labels_list = [], []
    search(directory, paths_list, labels_list)
    return paths_list, labels_list
       
        
def clean_dataset(dataset_dir, 
            remove_dirty_images=False, 
            copy_dirty_images=False, 
            visualizes_dirty_images=False, 
            print_dirty_images_info=False, 
            visualizes_number=40):
    """
    Inputs:
      - dataset_dir: The directory of the dataset.
      - remove_dirty_image: Whether to remove dirty data in the original directory.
      - copy_dirty_image: Whether to copy dirty data to a new directory.
      - visualizes_dirty_images: Whether to visualizes the dirty images.
      - print_dirty_images_info: Whether to print the information of the dirty images.
      - visualizes_number: The maximum number of dirty images visualization.
    """
    paths_list, labels_list = get_paths_and_labels_from_directory(dataset_dir)
    
    if copy_dirty_images:
        dirty_data_dir = os.path.dirname(dataset_dir) + '/' + os.path.basename(dataset_dir) + '_dirty'
        if not os.path.exists(dirty_data_dir):
            os.makedirs(dirty_data_dir)
    
    dirty_data = []
    for path, label in zip(paths_list, labels_list):
        if iu.check_image(path, color_mode='grayscale') is False:
            dirty_data.append((path, label))
            
    dirty_number = len(dirty_data)
    print('Found {} dirty images in the directory \'{}\'.'.format(dirty_number, dataset_dir))
    
    if dirty_number is not 0:
        if visualizes_dirty_images or print_dirty_images_info:
            iu.show_dirty_images(dirty_data, 
                          visualizes_dirty_images=visualizes_dirty_images, 
                          print_dirty_images_info=print_dirty_images_info, 
                          visualizes_number=visualizes_number)
            
    if copy_dirty_images or remove_dirty_images:
        for (path, label) in dirty_data:
            if copy_dirty_images:
                current_dirty_data_dir = join_path(dirty_data_dir, label) 
                if not os.path.exists(current_dirty_data_dir):
                    os.makedirs(current_dirty_data_dir)
                shutil.copy(path, current_dirty_data_dir)
            if remove_dirty_images:
                os.remove(path)     
    if copy_dirty_images:
        print('Copy {} dirty images to new directory \'{}\'.'.format(dirty_number, dirty_data_dir))
    if remove_dirty_images:
        print('Remove {} dirty images in the original directory \'{}\'.'.format(dirty_number, dataset_dir))


