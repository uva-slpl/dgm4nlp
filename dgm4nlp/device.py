"""
:Authors: - Wilker Aziz
"""


def tf_config(visible_device='1', allow_growth=True, mem_fraction=None, random_seed=None):
    import tensorflow as tf
    import os

    if random_seed is not None:
        import numpy as np
        tf.set_random_seed(random_seed)
        np.random.seed(random_seed)
    # TF setup for using GPU 0 and 70% of its memory
    if visible_device == '-1': # only CPU!
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_device
        return None
    else: # use a selected GPU
        config = tf.ConfigProto()
        if mem_fraction is not None:
            config.gpu_options.per_process_gpu_memory_fraction = mem_fraction
        else:
            config.gpu_options.allow_growth = allow_growth  # Allows dynamic mem allocation from minimum
        config.gpu_options.visible_device_list = visible_device
        return config


