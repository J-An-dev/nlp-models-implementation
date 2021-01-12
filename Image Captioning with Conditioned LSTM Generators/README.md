# Image Captioning with Conditioned LSTM Generators

## Introduction
This porject builds the following components: 

* Create matrices of image representations using an off-the-shelf image encoder.
* Read and preprocess the image captions. 
* Write a generator function that returns one training instance (input/output sequence pair) at a time. 
* Train an LSTM language generator on the caption data.
* Write a decoder function for the language generator. 
* Add the image input to write an LSTM caption generator. 
* Implement beam search for the image caption generator.

## Running on Google Colab
Due to the intense computation-power usage for training, recommend to run this Notebook on the Google Colab platform to take advantage of the GPU accelerator feature. Please follow the steps below:
* Step 1: Access your Colab account at https://colab.research.google.com/ and upload the `flick_caption_generator.ipynb` file by using selecting the 'upload' folder and then 'Choose file'
* Step 2: Click on 'Runtime' in the menu and then 'Change runtime type' and select 'GPU' as a value for 'Hardware accelerator'. Also make sure the checkbox 'Omit code cell output when saving this notebook' is *NOT* selected. You can also postpone this step until you actually need the GPU.