# -*- coding: utf-8 -*-


from recognition import data_utils as du
import csv
import pandas as pd
import numpy as np


def get_model_info(file_path):
    model_info_pd_data = pd.read_csv(file_path, sep=',',header=None)
    model_info = dict(model_info_pd_data.values)
    return model_info
    
    
def get_class_indices(file_path):
    """ Get the mapping from class names to class indices.
    
    Input:
     - file_path: A string contains the path of the file.
     
    Return:
     - class_indices: A dictionary contains the mapping from 
             classes to indices
    """
    class_indices_pd_data = pd.read_csv(file_path, sep=',',header=None)
    class_indices = dict(class_indices_pd_data.values)
    return class_indices


def get_top1_pd_data_from_topk(topk_path, top1_path):
    """Get the top-1 results in DataFrame format from the top-k results.
    
    Inputs: 
     - topk_path: A string contains the path of the top-k results.
     - top1_path: A string contains the path of the top-1 results.
     
   Return:
     - top1_df_data: Top-1 results in DataFrame format.
    """
    topk_df_data = pd.read_csv(topk_path)
    top1_df_data = topk_df_data.copy()
    top5_labels = topk_df_data['label'].values
    top1_labels = np.zeros_like(top5_labels, dtype=np.str)
    for index, top5 in enumerate(top5_labels):
        top1_labels[index] = top5[0]
    top1_df_data['label'] = top1_labels
    return top1_df_data
    
    
def get_topk_pd_data(test_generator, class_indices, 
              topk_indices=None, topk_classes=None):
    """Get top-k results in DataFrame format.
    
    Inputs:
      - test_generator: A data generator of the data set.
      - class_indices: A dictionary contains the mapping from class 
            names to class indices.
      - topk_indices: A list contains the top-k indices.
      
    Return:
      - topk_pd_data: Top-k results in DataFrame format. 
    """
    if topk_indices is None and topk_classes is None:
        raise ValueError('topk_indices and topk_classes can\'t both be \'None\'')
    topk_data = {}
    topk_data['filename'] = []
    topk_data['label'] = []
    if topk_indices is not None:
        for path, indices in zip(test_generator.filenames, topk_indices):
            classes = []
            for index in indices:
                # classes.append(list(class_indices)[index])
                classes.append(list(class_indices.keys())[index])
            classes = classes[::-1]
            file_name = du.get_file_name_from_path(path)
            topk_data['filename'].append(file_name)
            if type(classes[0]) is np.int64:
                topk_data['label'].append(','.join([str(each_class) for each_class in classes]))
            else:
                topk_data['label'].append(''.join(classes))
    
    elif topk_classes is not None:
        for path, classes in zip(test_generator.filenames, topk_classes):
            # classes.reverse()
            classes = classes[::-1]
            file_name = du.get_file_name_from_path(path)
            topk_data['filename'].append(filename)
            if type(classes[0]) is np.int64:
                topk_data['label'].append(','.join([str(each_class) for each_class in classes]))
            else:
                topk_data['label'].append(''.join(classes))
    topk_pd_data = pd.DataFrame(topk_data)
    return topk_pd_data

    
def get_pd_data_with_specific_order(csv_path, specific_order):
    """Get data in DataFrame format with specific order.
    
    Inputs:
     - csv_path: A string, the csv file's path.
     - specific_order: A string, the path of file which contains 
             the specific order.
     
    Return:
     - spo_pd_data: Data in DataFrame format with specific order.
    """
    data_dict = dict(pd.read_csv(csv_path).values)
    spo_data = {}
    spo_data['answer'] = []
    with open(specific_order) as f:
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            spo_data['answer'].append(line + ' ' + str(data_dict[line]))
    spo_pd_data = pd.DataFrame(spo_data) 
    return spo_pd_data
    
    
def write_list_or_dict_into_csv(data, csv_path, have_chinese=False):
    """Write a list or dict (e.g. class_indices) into a csv file.
    
    Inputs:
      - data: A list or dict.
      - csv_path: A string contains the path of the csv file.
      - have_chinese: True or False, whether the data has Chinese.
    """
    if type(data) not in {list, dict}:
        raise ValueError('The type of the input data must be \'dict\' or \'list\'')
    
    if have_chinese:
        encoding = 'utf-8-sig'
    else:
        encoding = 'utf-8'
        
    with open(csv_path, 'w', newline='', encoding=encoding) as f:
        w = csv.writer(f)
        if type(data) is dict:
            for key, val in data.items():
                w.writerow([key, val])
        else:
            w.writerows(data)
                
                
def write_pd_data_into_csv(pd_data, csv_path, have_chinese=False, index=True, header=True):
    """Write the result in DataFrame format into a csv file.
    
    Inputs:
      - pd_data: Data in DataFrame format.
      - csv_path: A string contains path of the csv file.
      - have_chinese: True or False, whether the data has Chinese.
      - index: True or False, whether write row names (index)
      - header: True or False, whether write out the column names. 
    """
    if have_chinese:
        encoding = 'utf-8-sig'
    else:
        encoding = 'utf-8'
    pd_data.to_csv(csv_path, encoding=encoding, index=index, header=header)
