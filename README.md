# Pneumonia Classification using Transfer Learning
In this project, [Xception](https://arxiv.org/abs/1610.02357) model is retrained to predict if patient has Pneumonia based on corresponding chest x ray. Dataset used for training and testing can be downloaded from [this](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) Kaggle link. 

## Motivation
Even after I read a few guides on how to detect Pneumonia, I wasnt able to predict if patient has Pneumonia for majority of x ray images. My motivation was to make a Transfer Learning model that will classify much better than I can.

## Data
Training set contains 5216 images. 1345 labeled with NORMAL, and 3875 labeled with PNEUMONIA.

Test set contains 624 images. 234 labeled with NORMAL, and 390 labeled with PNEUMONIA.

Images below are examples of healthy  lungs (left), and Pneumonia (right).

<img src="https://user-images.githubusercontent.com/43140432/68776245-10e71d00-0630-11ea-900a-ad4298bcc6f1.jpeg" title="Healhy lungs" width="250" height="250"> <img src="https://user-images.githubusercontent.com/43140432/68778091-f498af80-0632-11ea-9a8e-e6ed1c926f6a.jpeg" title="Pneumonia" width="250" height="250">

## Model
I have chosen Xception model for Transfer Learning because it had great results on Imagenet dataset.I added one Dense layer with two nodes (for two classes) on top of Xception model. This layer was trained together with the last 55 layers (parameter: retrain_layers) of the Xception network. I trained this model on my NVIDIA GeForce MX150 GPU. 

## Parameters
* output_classes = 2 - number of classes to classify
* learning_rate = 0.001 (default) - I couldnt find learning rate that works better.
* img_width, img_height, channel = 299, 299, 3 - required data shape for Xception model input.
* training_examples = 5216 - total number of training images.
* retrain_layers=100 - number of last Xception layer to be retrained.
* epochs = 10 - number of epochs that were required for a good results.
* batch_size = 10 - with larger size of retrain_layers batch_size is limited by my GPU memory (it has only 4096 MB maximum amount of memory). It is possible that training efficiency and results could be better if batch_size was higher.



