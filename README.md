这是我第一次参加与深度学习有关的比赛，参赛使用的是GTX1060的笔记本。由于机器性能的限制，我的模型在最终测试集的准确度是99.26。

# 项目介绍
[TinyMind第一届汉字书法识别挑战赛](https://www.tinymind.cn/competitions/41)

# 如何使用
- 首先你需要使用[clean_and_split_dataset.ipynb](https://github.com/finalacm/Chinese-character-recognition/blob/master/clean_and_split_dataset.ipynb)去获取你的训练集和验证集
- 这里我给出两个训练Demo：[base_pipeline_VGG16.ipynb](https://github.com/finalacm/Chinese-character-recognition/blob/master/base_pipeline_VGG16.ipynb)和[base_pipeline_ResNet50.ipynb](https://github.com/finalacm/Chinese-character-recognition/blob/master/base_pipeline_ResNet50.ipynb)的例子，你可以尝试训练你自己的网络
- 最后，如果你对单一模型的表现不满意，使用模型集成[ensemble.ipynb](https://github.com/finalacm/Chinese-character-recognition/blob/master/ensemble.ipynb)你可以得到更好的结果

# 程序介绍
csv_utils: csv模块
  1. get_model_info: 载入预训练模型和相应参数
  2. get_class_indices: 从csv文件中获取类名到索引值的字典
  3. get_top1_pd_data_from_topk: 从topk的csv文件中，以DataFrame格式获取top1并写入文件
  4. get_topk_pd_data: 以DataFrame格式获取topk结果
  5. get_pd_data_with_specific_order: 以特定文件顺序获取最终预测结果
  6. write_list_or_dict_into_csv: 将list或者dict类型的数据写入csv文件
  7. write_into_csv: 将DataFrame格式的数据写入csv文件

data_utils: 数据集处理模块
  1. rel_error: 相对错误
  2. join_path: 连接路径
  3. get_number_of_batches: 获取batch的个数
  4. get_file_name_from_path: 从路径中获取文件名
  5. get_class_name_from_path: 从路径中获取文件的类名
  6. split_dataset_into_training_and_test_sets: 划分数据集
  7. search: 搜索当前目录，并获取所有图像的路径和标签
  8. get_paths_and_labels_from_directory: 从目录下获取所有图像的路径和标签
  9. clean_dataset: 清洗数据集，可视化脏数据
 
evaluate_model: 评估模型模块
  1. predict_without_augmentation: 不通过数据增强来预测
  2. get_generators_for_models_list: 获取当前预训练模型的数据生成器
  3. save_prediction_for_models_list: 保存预训练模型的预测值
  4. get_predictions_from_backup_for_models_list: 获取备份后预训练模型的预测值
  5. get_topk_indices_by_single_generator: 通过单一数据生成器（模型）来预测前k大概率值所对应的索引值
  6. evaluate_topk_accuracy_by_single_generator: 通过单一数据生成器（模型）来预测top-k准确度
  7. get_topk_classes_by_single_generator: 通过单一数据生成器（模型）来获取前k大概率值所对应的类名
  8. predict_by_keras_augmentation: 通过KerasAPI自带的数据拓展进行预测集成 
  9. predict_by_my_augmentation: 用自己的方法对数据集进行拓展，再进行预测集成
  10. predictions_ensemble: 对多个模型给出的预测值进行集成
  11. get_generators_and_predictions_for_model_list: 获取多个预训练模型的数据生成器和预测值
  12. get_prediction_by_pre_trained_model: 通过预训练模型获取对数据集的预测值

image_utils: 图像处理模块
  1. cv_imread: OpenCV读取图像（可以包含中文路径）
  2. convert_RGB_to_BGR: 转换像素格式
  3. convert_BGR_to_RGB: 转换像素格式
  4. resize_image_by_size: 拉伸图像
  5. rotate_image_by_angle: 旋转图像
  6. check_image: 检查当前图像是否为脏数据（可以根据需要自行更改）
  7. show_dirty_images: 可视化脏数据
  8. show_images_in_wrong_info: 可视化CNN模型预测信息中的错误图像
  9. show_augment_patches: 可视化自己拓展后的图像
  10. show_image_by_keras_augmentation: 可视化Keras API数据增强后的图像

model_utils: 模型参数模块
  1. clear_session: 清除当前会话
  2. get_channels: 获取颜色通道数目
  3. print_model_info: 输出模型信息
  4. freeze_layers: 冻结CNN的一些网络层
  5. load_my_model: 载入模型和相应参数
  6. my_generator: 生成数据生成器
  7. draw_plot: 绘制训练历史曲线
  8. count_parameters: 统计模型的参数

train_model: 训练模型模块
  1. train_model: 训练模型

# 提示
- 数据增强以及抖动非常重要
- 你可以使用Batch NormalizationN和Dropout来提升你的模型
- 在一些全连接层增加正则化可以获得更好的结果
- 在过拟合的时候记得衰减学习率
- 不管怎么样，多尝试并从结果中学习是非常重要的


