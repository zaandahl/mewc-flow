import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D, SpatialDropout2D
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import keras_efficientnet_v2
from lib_data import create_dataset

def get_callback():
    cb = [EarlyStopping(monitor='loss', mode='min', min_delta=0.001, patience=3, restore_best_weights=True)]
    return(cb)

def get_base(
    mod,
    weights
):
    print(mod)
    print(weights)
    if mod == 'V1B0':
        model_base = keras_efficientnet_v2.EfficientNetV1B0(input_shape=(None,None,3), num_classes=0, pretrained=weights)
    elif mod == 'V1B1':
        model_base = keras_efficientnet_v2.EfficientNetV1B1(input_shape=(None,None,3), num_classes=0, pretrained=weights)
    elif mod == 'V1B2':
        model_base = keras_efficientnet_v2.EfficientNetV1B2(input_shape=(None,None,3), num_classes=0, pretrained=weights)
    elif mod == 'V2B0':
        model_base = keras_efficientnet_v2.EfficientNetV2B0(input_shape=(None,None,3), num_classes=0, pretrained=weights)
    elif mod == 'V2T':
        model_base = keras_efficientnet_v2.EfficientNetV2T(input_shape=(None,None,3), num_classes=0, pretrained=weights)
    elif mod == 'V2S':
        model_base = keras_efficientnet_v2.EfficientNetV2S(input_shape=(None,None,3), num_classes=0, pretrained=weights)
    elif mod == 'V2M':
        model_base = keras_efficientnet_v2.EfficientNetV2M(input_shape=(None,None,3), num_classes=0, pretrained=weights)
    else:
        print("Model not found. Falling back on V1B0.")
        model_base = keras_efficientnet_v2.EfficientNetV1B0(input_shape=(None,None,3), num_classes=0, pretrained=weights)
    return(model_base)

def build_model(
    num_classes,
    mod,
    weights,
    lr,
    dr
):
    if num_classes == 2:
        loss_f = 'binary_crossentropy'
        act_f = 'sigmoid'
        num_classes = 1
    else:
        #loss_f = 'categorical_crossentropy'
        loss_f = tfa.losses.SigmoidFocalCrossEntropy()
        act_f = 'softmax'
    model_base = get_base(mod, weights)
    # Freeze the pretrained weights
    model_base.trainable = False
    top_dropout_rate = dr
    model = Sequential()
    model.add(model_base)
    model.add(GlobalAveragePooling2D(name="avg_pool"))
    model.add(BatchNormalization(name="batch_norm"))
    model.add(Dropout(top_dropout_rate, name="top_dropout"))
    model.add(Dense(num_classes, activation=act_f, name="prediction")) 
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss_f, metrics=["accuracy"])    # tfa.losses.SigmoidFocalCrossEntropy()
    return model

def unfreeze_model(model, lr, num_classes):
    if num_classes == 2:
        loss_f = 'binary_crossentropy'
    else:
        loss_f = tfa.losses.SigmoidFocalCrossEntropy()
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss_f, metrics=["accuracy"])
    return(model)

# def exp_scheduler(epoch, lr): 
#     lr_base=0.256; decay_step=2.4; decay_rate=0.97; lr_min=0.00001; warmup=10
#     if epoch < warmup:
#         lr = (lr_base - lr_min) * (epoch + 1) / (warmup + 1)
#     else:
#         lr = lr_base * decay_rate ** ((epoch - warmup) / decay_step)
#         lr = lr if lr > lr_min else lr_min
#     return lr

def fit_frozen(
    model,
    num_classes,
    train_df,
    val_df,
    epochs,
    batch_size,
    target_shape,
    dropout=None,
    dropout_layer=-2,
    magnitude=1,
    out_lr=0.01
):
    cb = get_callback()
    train_ds = create_dataset(train_df, img_size=target_shape, batch_size=batch_size, magnitude=magnitude, ds_name="train")
    val_ds = create_dataset(val_df, img_size=target_shape, batch_size=batch_size, magnitude=magnitude, ds_name="validation")
    if dropout != None and isinstance(model.layers[dropout_layer], keras.layers.Dropout):
        print(">>>> Changing dropout rate to:", dropout)
        model.layers[dropout_layer].rate = dropout
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cb)
    unfreeze_model(model, lr=out_lr, num_classes=num_classes)
    return(hist, model)

def fit_progressive(
    model,
    train_df,
    val_df,
    lr_scheduler=None,
    total_epochs=36,
    batch_sizes=64,
    target_shapes=[128],
    dropouts=[0.4],
    dropout_layer=-2,
    magnitudes=[0],
):
    if model.compiled_loss is None:
        print(">>>> Error: Model NOT compiled.")
        return None
    histories = []
    stages = min([len(target_shapes), len(dropouts), len(magnitudes)])
    for stage, batch_size, target_shape, dropout, magnitude in zip(range(stages), batch_sizes, target_shapes, dropouts, magnitudes):
        print(">>>> stage: {}/{}, target_shape: {}, dropout: {}, magnitude: {}".format(stage + 1, stages, target_shape, dropout, magnitude))
        if len(dropouts) > 1 and isinstance(model.layers[dropout_layer], keras.layers.Dropout):
            print(">>>> Changing dropout rate to:", dropout)
            model.layers[dropout_layer].rate = dropout
        train_ds = create_dataset(train_df[stage], img_size=target_shape, batch_size=batch_size, magnitude=magnitude, ds_name="train")
        val_ds = create_dataset(val_df[stage], img_size=target_shape, batch_size=batch_size, magnitude=magnitude, ds_name="validation")
        initial_epoch = stage * total_epochs // stages
        epochs = (stage + 1) * total_epochs // stages
        history = model.fit(
            train_ds,  
            epochs=epochs,
            initial_epoch=initial_epoch,
            validation_data=val_ds,
            callbacks=[LearningRateScheduler(lr_scheduler)] if lr_scheduler is not None else [],
        )
        histories.append(history)
    hhs = {kk: np.ravel([hh.history[kk] for hh in histories]).astype("float").tolist() for kk in history.history.keys()}
    return(hhs, model)
