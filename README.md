# UW Neural Data Challenge 2019

The challenge was to predict the responses of visual neurons to given images.

Read more about this Kaggle challenge [here](https://www.kaggle.com/c/uwndc19).

## The Final Model Architecture

![Model Architecture](final_model.png)

The model receives an image of size 48x48x3 as input and returns 18 continuous 
values that correspond to a square root of a number of spikes for each neuron.

It was trained on 521 example, 20 examples were used for evaluation. During the training saturation of each 
image was changed every epoch by a random factor picked in the interval [0.7, 1.3]. Find more details 
about the training in the model's [configuration file](configs/multiclass/final-model.yaml).

## Some Thoughts

A model consisting of convolutional layers followed by dense layers was an obvious choice for this type of problem.
The main issue was a small dataset, and, as result, overfitting of training data.

#### Regularization

To reduce an overfitting, the model should be carefully regularized, but not too much, otherwise it will decrease
a model performance. I did a lot of experiments with dropouts, L2 regularizations and batch normalizations, 
and eventually ended up regularizing only dense layers using dropout.

#### Data Augmentation

Another technique to help a model generalize better is to get more data. Of course, we cannot get more real data, 
but we could try to generate it by distorting original images.

Introducing distortions to the images, there is a risk to remove some information that is crucial for the neurons 
spiking. So, we need to make an assumption that some distortions don't affect neurons perception and check this
hypothesis on evaluation data.

Unfortunately, the evaluation dataset is too small to say for sure that some distortions don't affect perception
and other do, but after some experiments I've concluded that the neurons don't care much about image 
saturation and rotations to small angles, but sensitive to vertical and horizontal flips.

The final model was trained on images that were modified every epoch by randomly changing their saturation. The 
second best model was trained with random saturation and, additionally, with rotations by a random angle in the 
interval [-0.1, 0.1] radians. I think, the rotation distortion didn't work very well, because the images are too 
small and interpolation after rotation actually changes some details on them.


## Model Configuration File

To train a new model you need to create a configuration file with hyper-parameters and training options. 
See a configuration example for the final model [here](configs/multiclass/final-model.yaml).

Here is the format of the configuration file:

```yaml
# input data configuration
data:
  # number of example use for evaluation
  eval_size: 20
  
  # a random seed for the dataset shuffling
  random_seed: 130
  
  # a list of random distortions that should be applied to
  # an image during training. Keys in this dictionary correspond
  # to functions from the uwndc19.helpers.image_distortion_funcs module
  distortions:
    # flips image horizontally
    flip_left_right: {}
    
    # flips image vertically
    flip_up_down: {}
    
    # chooses one the angles and rotates an image
    rotate_choice:
      angles: [0, 45, 90]
      
    # chooses a random angle from -N to N degrees and rotates an image,
    # uses one of two interpolations: BILINEAR (by default) or NEAREST
    rotate:
      max_angle: 30
      interpolation: BILINEAR

    # randomly changes an image brightness
    brightness:
      max_factor: 0.1
      
    # randomly changes image saturation
    saturation:
      min_value: 0.8
      max_value: 1.2

# model configuration
model:
  # an input image will be cropped to this width and height
  image_size: 48

  # number of neurons
  num_classes: 18

  # a list of convolution layers
  conv_layers:
    - num_filters: 32          # number of convolutional filters
      kernel_size: 5           # kernel size
      padding: same            # padding: "same" or "valid"
      dropout_rate: 0.0        # dropout rate (0 by default)
      l2_regularization: 0.0   # L2 regularization factor (0 by default)

  # a list of dense layers
  dense_layers:
    - num_units: 512           # number of units
      dropout_rate: 0.4        # dropout rate (0 by default)
      l2_regularization: 0.0   # L2 regularization factor (0 by default)

  # a final logits layer
  logits_layer:
    dropout_rate: 0.4          # dropout rate (0 by default)
    l2_regularization: 0.0     # L2 regularization factor (0 by default)

# training configuration
training:
  eval_steps: 50               # do evaluation every N steps
  early_stopping_evals: 5      # stop training if RMSE was not improved after N evaluations
  learning_rate: 0.001         # learning rate
  keep_checkpoint_max: 3       # maximum number of checkpoints to keep
  export_best_models: true     # export models with the best RMSE
  exports_to_keep: 3           # number of the best models to keep
```
