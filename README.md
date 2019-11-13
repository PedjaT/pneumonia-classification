## Pneumonia Classification using Transfer Learning
In this project, [Xception](https://arxiv.org/abs/1610.02357) model is retrained to predict if patient has Pneumonia based on corresponding chest x ray. Dataset used for training and testing can be downloaded from [this](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) Kaggle link. 

# Motivation
Even after I read a few guides on how to detect Pneumonia, I wasnt able to predict if patient has Pneumonia for majority of x ray images. My motivation was to make a Transfer Learning model that will much classify better than I can.

# Data
Training set contains 5216 images. 1345 labeled with NORMAL, and 3875 labeled with PNEUMONIA.
Test set contains 624 images. 234 labeled with NORMAL, and 390 labeled with PNEUMONIA.
Images below are examples of healthy  lungs (left), and Pneumonia (right).
<img src="https://user-images.githubusercontent.com/43140432/68776245-10e71d00-0630-11ea-900a-ad4298bcc6f1.jpeg" title="Healhy lungs" width="250" height="250"> <img src="https://user-images.githubusercontent.com/43140432/68778091-f498af80-0632-11ea-9a8e-e6ed1c926f6a.jpeg" title="Pneumonia" width="250" height="250">
