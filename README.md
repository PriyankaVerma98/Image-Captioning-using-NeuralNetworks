# Image Captioning using Neural_networks

### Aim : 
 - To design and implement image and video caption generation deep neural network architecture.

### Set Up

 - Python SciPy environment, ideally with Python 3.
 - Keras (2.2 or higher) is installed with the TensorFlow
 - Libraries such as scikit-learn, Pandas, NumPy, and Matplotlib 
 - GPU (accessible through Google Collaboratories). Following libraries are imported to map GPU requirements
    ```python
     import​ psutil 
     import​ humanize 
     import​ os
     import​ GPUtil ​as​ GPU
     ```
     
### Data

**Flickr8k dataset**

It consists of 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events. The reason to select is because it is realistic and relatively small to download it and build models using a Gogle Colab GPU.
    - Flickr8k_Dataset.zip​ (1 Gigabyte) An archive of all photographs.
    - Flickr8k_text.zip​ (2.2 Megabytes) An archive of all text descriptions for
photographs.

### Methodology

- Photo Feature Extractor : using the 16- layered VGG model pre-trained on the ImageNet dataset
- Sequence Processor : this is a word embedding layer for handling the text input, followed by a Long Short-Term Memory (LSTM) recurrent neural network layer
- 

model summary    
---
This was my first hands on project in Neural Networks and was done as part of my Laboratory Project for course CS F366 at BITS Pilani.
