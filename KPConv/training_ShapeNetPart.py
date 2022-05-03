
#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on ShapeNetPart dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Common libs
import time
import os
import sys
#######################
#      ATTENTION      #
#######################
# UNCOMMENT BELOW IF YOU ARTHORIZED TO TRACK THE EXPERIMENT WITH WANDB.AI
#import wandb
#wandb.login(key='')
#wandb.init(project="")

import tensorflow as tf
# Custom libs
from utils.config import Config
from utils.trainer import ModelTrainer
from models.KPFCNN_model import KernelPointFCNN

# Dataset
from datasets.ShapeNetPart import ShapeNetPartDataset


# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#


class ShapeNetPartConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """
    #config = wandb.config

    ####################
    # Dataset parameters
    ####################

    # Dataset name in the format 'ShapeNetPart_Object' to segment an object class independently or 'ShapeNetPart_multi'
    # to segment all objects with a single model.
    dataset = 'ShapeNetPart_Pole'

    # Number of classes in the dataset (This value is overwritten by dataset class when initiating input pipeline).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    network_model = None

    # Number of CPU threads for the input pipeline
    input_threads = 8
    
    # Used features
    color_info = True
    intensity_info = True
    #########################
    # Architecture definition
    #########################

    # Define layers
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb_deformable',
                    'resnetb_deformable_strided',
                    'resnetb_deformable',
                    'resnetb_deformable_strided',
                    'resnetb_deformable',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']

    # KPConv specific parameters
    #num_kernel_points = config.num_kernel_points
    num_kernel_points = 15 
    first_subsampling_dl = 0.1

    # Density of neighborhoods for deformable convs (which need bigger radiuses). For normal conv we use KP_extent
    #density_parameter = config.density_parameter 
    density_parameter = 5.000

    # Influence function of KPConv in ('constant', 'linear', gaussian)
    #KP_influence = config.KP_influence
    KP_influence = 'linear'
    #KP_extent = config.KP_extent
    KP_extent = 1

    # Aggregation function of KPConv in ('closest', 'sum')
    #convolution_mode = config.convolution_mode
    convolution_mode  = 'sum'

    # Can the network learn modulations in addition to deformations
    modulated = False

    # Offset loss
    # 'permissive' only constrains offsets inside the big radius
    # 'fitting' helps deformed kernels to adapt to the geometry by penalizing distance to input points
    #offsets_loss = config.offsets_loss
    offsets_loss = 'fitting' 
    offsets_decay = 0.1

    # Choice of input features
    in_features_dim = 7

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.98

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 1000

    # Learning rate management
    #learning_rate = config.learning_rate
    learning_rate = 0.01
    momentum = 0.98
    lr_decays = {i: 0.1**(1/80) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of batch
    #batch_num = config.batch_num
    batch_num = 20

    # Number of steps per epochs (cannot be None for this dataset)
    epoch_steps = None

    # Number of validation examples per epoch
    validation_size = 50

    # Number of epoch between each snapshot
    snapshot_gap = 50

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [False, False, False]
    augment_rotation = 'none'
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_noise = 0.001
    augment_occlusion = 'none'

    # Whether to use loss averaged on all points, or averaged per batch.
    batch_averaged_loss = False

    # Do we nee to save convergence
    saving = True
    saving_path = None


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':

    ##########################
    # Initiate the environment
    ##########################

    # Choose which gpu to use
    GPU_ID = '1'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    # Enable/Disable warnings (set level to '0'/'3')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

    ###########################
    # Load the model parameters
    ###########################
   
    #config = wandb.config
    config = ShapeNetPartConfig()
   
    ##############
    # Prepare Data
    ##############

    print()
    print('Dataset Preparation')
    print('*******************')

    # Initiate dataset configuration
    dataset = ShapeNetPartDataset(config.dataset.split('_')[1], config.input_threads)

    # Create subsampled input clouds
    dl0 = config.first_subsampling_dl
    color_info = config.color_info
    intensity_info = config.intensity_info
    dataset.load_subsampled_clouds(dl0, color_info, intensity_info)

    # Initialize input pipelines
    dataset.init_input_pipeline(config)

    # Test the input pipeline alone with this debug function
    # dataset.check_input_pipeline_timing(config)

    ##############
    # Define Model
    ##############

    print('Creating Model')
    print('**************\n')
    t1 = time.time()

    # Model class
    model = KernelPointFCNN(dataset.flat_inputs, config)

    # Trainer class
    trainer = ModelTrainer(model)
    t2 = time.time()

    print('\n----------------')
    print('Done in {:.1f} s'.format(t2 - t1))
    print('----------------\n')

    ################
    # Start training
    ################

    print('Start Training')
    print('**************\n')

    trainer.train(model, dataset)



