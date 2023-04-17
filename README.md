# Stabilized Regularized Neural Networks (StReNN)

GitHub repository to reproduce experiments from the paper: *To be or not to be stable, that is the question: understanding neural networks for inverse problems*. The paper is now a pre-print on arXiv. Please refer to https://arxiv.org/abs/2211.13692.

## Installation
The use the code, simply clone the GitHub repository locally with

```
git clone https://github.com/loibo/stabilize_neural_networks.git
```

Moreover, the `IPPy` library is required to execute portions of the code. `IPPy` contains utility functions to work with reconstructors. It can be accessed by cloning the `IPPy` repository locally, inside of the main folder of this project. To do that, simply move inside of the project directory and run

```
git clone https://github.com/devangelista2/IPPy.git
```

## Project Stucture
Please note that the functions requires a specific folders and files structure to work. Since, due to memory constraint, it was not possible to upload the whole project on GitHub, the user is asked to create some folders to follow the required structure. A diagram representing it is given in the following.

```
|- data
|   |- GOPRO_train_small.npy
|   |- GOPRO_test_small.npy
|- images
|   |- ...
|- IPPy
|   |- ...
|- model_weights
|   |- ...
|- plots
|   |- ...
|- results
|   |- ...
|- statistics
|   |- ...
|- other_files.py
```

This can be obtained by simply creating the `data` and the `model_weights` folders by running:

```
mkdir data
mkdir model_weights
```

For informations about how to download the data (to be placed inside the `data` folder), and the pre-trained model weights, please refer to the following.

## Datasets
To run the experiments, the training and the test set has to be downloaded. A copy of the data used to train the models and get the results for the paper is available on HuggingFace. To get it, simply create a folder named `data` into the main project directory, move into that and run the following command:

```
git lfs install
git clone https://huggingface.co/datasets/TivoGatto/celeba_grayscale
```

which will download the data, in `.npy` format, used in the experiments. It is a slighly modified version of the GoPro dataset, where the images has been cropped to be $256 \times 256$ images, and converted to grey-scale with standard conversion algorithms. 

## Pre-trained models

The pre-trained models will be available soon.

## BibTex citation
Consider citing our work if you use it. Here the Bibtex for the citation.

```
@article{evangelista2022or,
  title={To be or not to be stable, that is the question: understanding neural networks for inverse problems},
  author={Evangelista, Davide and Nagy, James and Morotti, Elena and Piccolomini, Elena Loli},
  journal={arXiv preprint arXiv:2211.13692},
  year={2022}
}
```