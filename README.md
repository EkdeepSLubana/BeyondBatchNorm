# Beyond BatchNorm: Towards a Unified Understanding of Normalization in Deep Learning

Codebase for the paper ["Beyond BatchNorm: Towards a Unified Understanding of Normalization in Deep Learning."](https://arxiv.org/abs/2106.05956)

## Requirements

The code requires:

* Python 3.6 or higher

* Pytorch 1.7 or higher

To install other dependencies, the following command can be used (uses pip):

```setup
./requirements.sh
```

## Organization
The provided modules serve the following purpose:

* **main.py**: Provides functions for training models with different layers.

* **layer_defs.py**: Contains definitions for different normalization layers. 

* **models.py**: Contains definitions for different model architectures.

* **config.py**: Training hyperparameters and progress bar definition.

## Example execution 
To train a model (e.g., ResNet-56) using a particular normalization layer (e.g., BatchNorm), run the following command

```execution
python main.py -arch=ResNet-56 --norm_type=BatchNorm
```

## Summary of basic options

```--arch=<architecture> ```

- *Options*: vgg / resnet-56. 
- Since our non-residual CNNs are like VGG, we refer to their architecture as VGG.

```--p_grouping=<amount_of_grouping_in_GroupNorm> ```

- *Options*: integer; *default*: 32. 
- If p_grouping < 1: defines a group size of 1/p_grouping. E.g., p_grouping=0.5 implies group size of 2. 
- If p_grouping >= 1: defines number of groups as layer_width/p_grouping. E.g., p_grouping=32 implies number of groups per layer will be 32.

```--skipinit=<use_skipinit_initialization> ```

- *Options*: True/False; *Default*: False. 

```--preact=<use_preactivation_resnet> ```

- *Options*: True/False; *Default*: False. 

```--probe_layers=<probe_activations_and_gradients> ```

- *Options*: True/False; *Default*: True
- Different properties in model layers (activation norm, stable rank, std. dev., cosine similarity, and gradient norm) will be calculated every iteration and stored as a dict every 5 epochs of training

```--init_lr=<init_lr> ```

- *Options*: float; *Default*: 1. 
- A multiplication factor to alter the learning rate schedule (e.g., if default learning rate is 0.1, init_lr=0.1 will make initial learning rate be equal to 0.01).

```--lr_warmup=<lr_warmup> ```

- *Options*: True/False; *Default*: False.
- Learning rate warmup; used in Filter Response Normalization.

```--batch_size=<batch_size> ```

- *Options*: integer; *Default*: 256. 

```--dataset=<dataset> ```

- *Options*: CIFAR-10/CIFAR-100; *Default*: CIFAR-100.

```--download=<download_dataset> ```

- *Options*: True/False; *Default*: False.
- If CIFAR-10 or CIFAR-100 are to be downloaded, this option should be True.

```--cfg=<number_of_layers> ```

- *Options*: cfg_10/cfg_20/cfg_40; *Default*: cfg_10
- Number of layers for non-residual architectures.

```--seed=<change_random_seed> ```

- *Options*: integer; *Default*: 0.

**Training Settings**: To change number of epochs or the learning rate schedule for training, change the hyperparameters in *config.py*. By default, models are trained using SGD with momentum (0.9).
