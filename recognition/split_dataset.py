# -*- coding: utf-8 -*-


import numpy as np
import os
import shutil

def split_dataset_into_training_and_test_sets(data_dir, training_dir, 
                               test_dir, test_size=0.1):
    """
    Inputs:
      - data_dir: The directory of the source data
          the structure must be:
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
      - training_dir: The directory of the training set
      - test_dir: The directory of the test set
      - test_size: A real number between 0 and 1, represent the 
          proportion of the dataset to include in the train split
      
    Returns:
      - training_samples: Total number of the training samples
      - test_samples: Total number of the test samples
    """
    if test_size<0 or test_size>1:
        raise ValueError('Test size must be a real number between 0 and 1.')
        
    if not os.path.exists(data_dir):
        raise ValueError('The data directory {} does not exist!'.format(data_dir))
        
    if os.path.exists(training_dir):
        shutil.rmtree(training_dir, ignore_errors=False)
    os.makedirs(training_dir)
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir, ignore_errors=False) 
    os.makedirs(test_dir)
    
    training_set_files = 0
    test_set_files = 0
    
    classes = []
    training_set = []
    test_set = []
    
    for root, subdirs, files in os.walk(data_dir):
        class_name = os.path.basename(root)
        if class_name == root:
            continue
        
        training_class_dir = os.path.join(training_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
                        
        if not os.path.exists(training_class_dir):
            os.makedirs(training_class_dir)
                
        if not os.path.exists(test_class_dir):
            os.makedirs(test_class_dir)
        
        total_files = len(files)
        mark = np.ones((total_files, ), dtype=int)
        mask = np.random.choice(total_files, int(total_files*test_size))
        mark[mask] = 0
        
        class_training_files = 0
        class_test_files = 0
        for index, file in enumerate(files):
            file_path = os.path.join(root, file)
            if mark[index] == 1:
                shutil.copy(file_path, training_class_dir + '/' + file)
                class_training_files += 1
            else:
                shutil.copy(file_path, test_class_dir + '/' + file)
                class_test_files += 1
                
        training_set_files += class_training_files
        test_set_files += class_test_files
        classes.append(class_name)
        training_set.append(class_training_files)
        test_set.append(class_test_files)
        
    print('The number of files in the source data set: {}'.format(
        training_set_files + test_set_files))
    print('The number of files in the training set: {}'.format(training_set_files))
    print('The number of files in the test set: {}'.format(test_set_files))
    
    print('All the classes in the source data set:')
    print(classes)
    print('The number of files for each class in the training set:')
    print(training_set)
    print('The number of files for each class in the test set:')
    print(test_set)
    
    return training_set_files, test_set_files