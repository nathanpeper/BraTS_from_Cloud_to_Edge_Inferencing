import tensorflow as tf   # TensorFlow 2
from tensorflow import keras as K

import os
import datetime
from pathlib import Path

from argparser import args
from dataloader import DatasetGenerator
from model import dice_coef, soft_dice_coef, dice_loss, unet_3d


def test_intel_tensorflow(attempt_enable=True):
    """
    Check if Intel OneDNN optimizations for TensorFlow are enabled
    """
    if attempt_enable==True:
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

    import tensorflow as tf

    print("We are using Tensorflow version {}".format(tf.__version__))

    mkl_enabled = False
    major_version = int(tf.__version__.split(".")[0])
    minor_version = int(tf.__version__.split(".")[1])
    if major_version >= 2:
        if minor_version < 5:
            from tensorflow.python import _pywrap_util_port
        elif minor_version >= 9:

            from tensorflow.python.util import _pywrap_util_port
            onednn_enabled = int(os.environ.get('TF_ENABLE_ONEDNN_OPTS', '1'))

        else:
            from tensorflow.python.util import _pywrap_util_port
            onednn_enabled = int(os.environ.get('TF_ENABLE_ONEDNN_OPTS', '0'))
        mkl_enabled = _pywrap_util_port.IsMklEnabled() or (onednn_enabled == 1)
    else:
        mkl_enabled = tf.pywrap_tensorflow.IsMklEnabled()
   
    print(f"Intel-optimizations (DNNL) enabled: {mkl_enabled}"
    

print(args)
test_intel_tensorflow()  # Prints if Intel-optimized TensorFlow is used.

# """
# crop_dim = Dimensions to crop the input tensor
# """
# crop_dim = (args.tile_height, args.tile_width,
#             args.tile_depth, args.number_input_channels)

"""
1. Load the dataset
"""
brats_data = DatasetGenerator(data_dir=args.data_dir,
                 dataset=args.dataset,
                 data_path=args.data_path,
                 train_test_split=args.train_test_split,
                 validate_test_split=args.validate_test_split,
                 batch_size_train=args.batch_size_train,
                 batch_size_validate=args.batch_size_validate,
                 batch_size_test=args.batch_size_test,
                 tile_height=args.tile_height,
                 tile_width=args.tile_width,
                 tile_depth=args.tile_depth,
                 number_input_channels=args.number_input_channels,
                 crop_dim = (args.tile_height, args.tile_width,
                             args.tile_depth, args.number_input_channels),
                 number_output_classes=args.number_output_classes,
                 random_seed=args.random_seed)

brats_data.print_info()  # Print dataset information

"""
2. Create the TensorFlow model
"""
model = unet_3d(input_dim=(args.tile_height, args.tile_width, args.tile_depth, args.number_input_channels),  
                filters=args.filters,
                number_output_classes=args.number_output_classes,
                use_upsampling=args.use_upsampling,
                concat_axis=-1, 
                model_name=args.saved_model_name)

local_opt = K.optimizers.Adam()
model.compile(loss=dice_loss,
              metrics=[dice_coef, soft_dice_coef],
              optimizer=local_opt)

saved_model_path = Path(args.model_dir / args.saved_model_name)          
checkpoint = K.callbacks.ModelCheckpoint(saved_model_path,
                                         verbose=1,
                                         save_best_only=True)

# TensorBoard
logs_dir = os.path.join(
    "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_logs = K.callbacks.TensorBoard(log_dir=logs_dir)

callbacks = [checkpoint, tb_logs]

"""
3. Train the model
"""
steps_per_epoch = brats_data.num_train // args.batch_size
model.fit(brats_data.get_train(), epochs=args.epochs,
          steps_per_epoch=steps_per_epoch,
          validation_data=brats_data.get_validate(),
          callbacks=callbacks,
          verbose=1)

"""
4. Load best model on validation dataset and run on the test
dataset to show generalizability
"""
best_model = K.models.load_model(saved_model_path,
                                 custom_objects={"dice_loss": dice_loss,
                                                 "dice_coef": dice_coef,
                                                 "soft_dice_coef": soft_dice_coef})

print("\n\nEvaluating best model on the testing dataset.")
print("=============================================")
loss, dice_coef, soft_dice_coef = best_model.evaluate(brats_data.get_test())

print("Average Dice Coefficient on testing dataset = {:.4f}".format(dice_coef))

"""
5. Save the best model without the custom objects (dice, etc.)
   NOTE: You should be able to do .load_model(compile=False), but this
   appears to currently be broken in TF2. To compensate, we're
   just going to re-compile the model without the custom objects and
   save as a new model (with suffix "_final")
"""
final_model_name = args.saved_model_name + "_final"

# best_model.compile(loss="binary_crossentropy", metrics=["accuracy"],
#                    optimizer="adam")

final_model_path = Path(args.model_dir / final_model_name)
K.models.save_model(best_model, final_model_path,
                    include_optimizer=False)
