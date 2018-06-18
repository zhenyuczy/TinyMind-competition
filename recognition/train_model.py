# -*- coding: utf-8 -*-

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, LambdaCallback, ReduceLROnPlateau
from keras.layers import BatchNormalization, Activation, Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU, PReLU
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
    

def train_model(model, train_generator, valid_generator, model_path, **kwargs):
    """ Train and save the best model
    
    Inputs:
      - model:
      - train_generator: The generator of the training set
      - valid_generator: The generator of the validation set
      - model_path: The path of the best model.
      - kwargs:
          - print_lr: Decide whether to print the learning rate after each epoches end.
          - batch_size: 
          - learning_rate: 
          - decay: 
          - epochs:
          - automatic_reduction: True or False. 
              If True, then those parameters are . Otherwise, .
              - patience: 
              - reduce_factor: An float, decaying factor of learning rate
              - reduce_time: An integer, number of decaying learning rate
              - reduce_monitor: 'acc', 'loss', 'val_acc', 'val_loss'. If the parameter 
                  does not improve, then reduce learning rate.
          - save_best: 'acc', 'loss', 'val_acc', 'val_loss', save the best model's 
              weights.
          
    Return:
      - history: A dictionary contains the training phase information
    """
    print_lr = kwargs.get('print_lr', False)
    batch_size = kwargs.get('batch_size', 32)
    learning_rate = kwargs.get('learning_rate', 0.01)
    decay = kwargs.get('decay', 0)
    epochs = kwargs.get('epochs', 40)
    patience = kwargs.get('patience', 5)
    reduce_factor = kwargs.get('reduce_factor', 0.1)
    reduce_time = kwargs.get('reduce_time', 3)
    reduce_monitor = kwargs.get('reduce_moitor', 'val_acc')
    save_best = kwargs.get('save_best', 'val_acc')
    automatic_reduction = kwargs.get('automatic_reduction', False)
    
    min_lr = learning_rate * (10**(-reduce_time))
    # reduce learning rate
    reduce_lr = ReduceLROnPlateau(monitor=reduce_monitor, factor=reduce_factor,
                        patience=patience, min_lr=min_lr)
    
    mdcheck = ModelCheckpoint(filepath=model_path, 
                     monitor=save_best, save_best_only=True)
    
    if print_lr:
        # print learning rate after each epoch
        lr_print_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: print(
            ' - lr: {}'.format(K.eval(model.optimizer.lr))))
        
        if automatic_reduction:
            callbacks = [mdcheck, reduce_lr, lr_print_callback]
        else:
            callbacks = [mdcheck, lr_print_callback]
    else:
        if automatic_reduction:
            callbacks = [mdcheck, reduce_lr]
        else:
            callbacks = [mdcheck]
        
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