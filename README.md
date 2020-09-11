# OrthoReg: Robust Network Pruning Using Orthogonality Regularization
Code release for OrthoReg. The main paper can be found at https://arxiv.org/abs/2009.05014.

## Brief Summary
OrthoReg is a regularization strategy aimed at making importance estimates in prior state-of-the-art pruning methods based on loss preservation (e.g., see [Fisher pruning](https://arxiv.org/pdf/1801.05787.pdf)) more reliable and robust for large pruning ratios. We provide code for extracting **Early-Bird Tickets** from and for **Iteratively Pruning** VGG-13, MobileNet-V1, and ResNet-34.

<img src="image/OrthoReg.png" height="150">

The code requires:

* Python 3.6 or higher

* Pytorch 1.4 or higher

## Requirements

To install requirements (uses pip):

```setup
./requirements.sh
```

## Organization
The provided modules serve the following purpose:

* **main.py**: Provides functions for training pruned networks in general, including Early-Bird Tickets.

* **eval.py**: Calculate train accuracy, test accuracy, degree of orthogonality (layerwise), FLOPs, and compression ratio.

* **imp_estimator.py**: Importance estimators for different methods ([Fisher](https://arxiv.org/pdf/1801.05787.pdf), [BN](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf), [TFO](https://arxiv.org/pdf/1611.06440.pdf), [GraSP](https://openreview.net/forum?id=SkgsACVKPH), [RDT](http://proceedings.mlr.press/v97/gao19c/gao19c.pdf), [L1](https://openreview.net/pdf?id=rJqFGTslg), and [SFP](https://arxiv.org/pdf/1808.06866.pdf)).

* **pruner.py**: Pruning engine (includes pruned networks' classes).

* **models.py**: Model classes for VGG-13, MobileNet-V1, ResNet-34.

* **config.py**: Hyperparameters and progress bar for training models.

Pretrained models are to be stored in the directory **pretrained** and pruned models will be saved in **pruned_nets**. We provide both minimally and fully trained models for Early-Bird Tickets and iterative pruning experiments, respectively (see below). 

## Extracting Early-Bird Tickets
To extract Early-Bird Tickets from a model (e.g., VGG-13) on CIFAR-100, run the following command

```ebt
python main.py --ebt=True --model=vgg --pretrained=False --pruning_type=orthoreg --prune_percent=25
```

## Iterative Pruning 
To extract Early-Bird Tickets from a model (e.g., VGG-13) on CIFAR-100, run the following command

```iterative
python main.py --model=vgg --pretrained=False --pruning_type=orthoreg --prune_percent=50 --n_rounds=2
```

## Summary of available options

```--ebt=<extract_early_bird_tickets> ```

- *Options*: True/False; default: False. 
- If False, n_rounds (see below) needs to be specified to prune iteratively.

```--model=<model_name> ```

- *Options*: vgg/mobilenet/resnet. 

```--pretrained=<use_pretrained_model> ```

- *Options*: True/False. 
- For True, pretrained models should be stored in the directory **pretrained**.

```--data_path=<path_to_data> ```

- *Options*: string; *Default*: "CIFAR100". 
- If CIFAR-100 is not already downloaded, it will be downloaded.
- If you are using your own dataset, indicate the path to the main dataset directory which should contain two subdirectories: train and test.

```--pruning_type=<how_to_estimate_importance> ```

- *Options*: bn/fisher/orthoreg/tfo/l1/sfp/rdt
- bn is BN-scale based importance, as used by the [original work on extracting Early-Bird Tickets](https://openreview.net/pdf?id=BJxsrgStvr).

```--pruning_percent=<how_much_percent_filters_to_prune> ```

- *Options*: integer in range [0, 95]. 

```--n_rounds=<number_of_pruning_rounds> ```

- *Options*: integer. 

```--thresholds=<manual_thresholds_for_pruning> ```

- *Options*: 'array'. 
- If you do not want to use the default method for deciding pruning ratios, use this option to define manual thresholds.
- E.g., for pruning the network by 10%, 50%, 80% of original filters in 3 rounds, respectively, use --thresholds='[10, 50, 80]'

```--seed=<change_random_seed> ```

- *Options*: integer; *Default*: 0.

```--only_train=<only_train_do_not_prune> ```

- *Options*: base/ortho/False; *Default*: False.

**Training Settings**: To change number of epochs or the learning rate schedule for training the base models or the pruned models, change the hyperparameters in *config.py*. By default, models are trained using SGD with momentum (0.9).

## Evaluation

To evaluate a model (e.g., a pruned VGG-13 model), use:

```eval
python eval.py --model vgg --pruned True --model_path <path_to_model_file> --test_acc True
```

Summary of available options for evaluating models:

```--model=<model_name> ```

- *Options*: vgg/mobilenet/resnet. 

```--pruned=<evaluating_a_pruned_model> ```

- *Options*: True/False; *Default*: False. 
- Set to True for evaluating a pruned model.

```--model_path=<path_to_model> ```

- *Options*: string. 
- Location where model to be analyzed is stored.

```--data_path=<path_to_dataset> ```

- *Options*: string; *Default*: "CIFAR100". 
- If CIFAR-100 is not already downloaded, it will be downloaded.
- If you are using your own dataset, indicate the path to the main dataset directory which should contain two subdirectories: train and test.

```--train_acc=<evaluate_train_accuracy> ```

- *Options*: True/False; *Default*: False.

```--test_acc=<evaluate_test_accuracy> ```

- *Options*: True/False; *Default*: False.

```--flops=<evaluate_flops_in_model> ```

- *Options*: True/False; *Default*: False.

```--compression=<evaluate_compression_ratio> ```

- *Options*: True/False; *Default*: False.

```--eval_ortho=<evaluate_degree_of_orthogonality> ```

- *Options*: True/False; *Default*: False.
- For layer *l* with weight matrix **W(l)** , the degree of Orthogonality option outputs the ratio ||diag(**W(l)**^T **W(l)**)|| / ||**W(l)**^T **W(l)**||. For a diagonal matrix, this ratio should be 1.

<br>

## Results 

We provide sample results for our code. Following are the performances of the Early-Bird Tickets drawn using different pruning methods (OrthoReg, Fisher pruning, and BN-scale based pruning) on VGG-13, MobileNet-V1, and ResNet-34 models for the CIFAR-100 dataset:

To replicate these, use seed 0.

* Original models:

    |     Model name     | Accuracy |
    | ------------------ |----------|
    |     *ResNet-34*    |   73.4%  |
    |      *VGG-13*      |   65.5%  |
    |   *MobileNet-V1*   |   67.0%  |

* OrthoReg:

    |     Model name     | % Pruned | Accuracy |     Model name     | % Pruned | Accuracy |     Model name     | % Pruned | Accuracy |
    | ------------------ |----------| -------- | ------------------ |----------| -------- | ------------------ |----------| -------- |
    |     *ResNet-34*    |    25%   |   77.4%  |      *VGG-13*      |    25%   |   71.4%  |   *MobileNet-V1*   |    25%   |   67.8%  |
    |                    |    50%   |   76.7%  |                    |    50%   |   71.2%  |                    |    50%   |   67.4%  |
    |                    |    75%   |   74.8%  |                    |    75%   |   67.5%  |                    |    75%   |   65.8%  |

* Fisher pruning:

    |     Model name     | % Pruned | Accuracy |     Model name     | % Pruned | Accuracy |     Model name     | % Pruned | Accuracy |
    | ------------------ |----------| -------- | ------------------ |----------| -------- | ------------------ |----------| -------- |
    |     *ResNet-34*    |    25%   |   72.7%  |      *VGG-13*      |    25%   |   65.4%  |   *MobileNet-V1*   |    25%   |   67.9%  |
    |                    |    50%   |   72.3%  |                    |    50%   |   67.1%  |                    |    50%   |   67.6%  |
    |                    |    75%   |   71.3%  |                    |    75%   |   65.3%  |                    |    75%   |   65.9%  |

* BN-scale based:

    |     Model name     | % Pruned | Accuracy |     Model name     | % Pruned | Accuracy |     Model name     | % Pruned | Accuracy |
    | ------------------ |----------| -------- | ------------------ |----------| -------- | ------------------ |----------| -------- |
    |     *ResNet-34*    |    25%   |   72.9%  |      *VGG-13*      |    25%   |   65.5%  |   *MobileNet-V1*   |    25%   |   67.7%  |
    |                    |    50%   |   72.1%  |                    |    50%   |   65.6%  |                    |    50%   |   68.3%  |
    |                    |    75%   |   70.1%  |                    |    75%   |   64.8%  |                    |    75%   |   65.5%  |

## Pre-trained Models

We provide fully trained and minimally trained models that can be pruned using our code. 

For Early-Bird Tickets experiments, minimally trained models can be found here: 

- [Minimally trained: CIFAR-100](https://drive.google.com/drive/folders/1HjP2me8G4vD3z-7QkPZRxoMmmATbEqVx?usp=sharing). 

For iterative pruning, fully trained models can be found here:

- [Iterative Pruning: CIFAR-100](https://drive.google.com/drive/folders/1VwME9d-ZZUkksM-eoh5rwmGK_b6YWr7Y?usp=sharing).

- [Iterative Pruning: Tiny-ImageNet](https://drive.google.com/drive/folders/1zmq8UeuKHPJYSW-m3DlmBD39080BhMKD?usp=sharing).

To use these models:

- Store the fully trained models in the directory pretrained/iterative.

- Store the minimally trained models in the directory pretrained/ebt.

### Note
- The optimizer is set to SGD by default. For iterative pruning experiments described in the paper, Adam was used.
