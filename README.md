# ImageAnalysis
The programs herein were originally developed to analyze wheat seeds severity for fusarium damaged kernels. The python program detects and extract data RGB, area, and length information from each seed in an image while the R program utilize the extracted information to predict the disease score of the image.

# EfficientNet (Update)

The repository hosts a REST API built using FastAPI to serve a trained EfficientNet B2 model for predicting disease scores in wheat images. The API allows users to upload images and receive predictions of disease scores in real-time. This provides a significant improvement over previous model with the following features:

- **EfficientNet B2 Integration**: Pre-trained on several wheat images scored or fusarium damaged kernels. 
- **Real-Time Inference**: Provides predictions in real time.
- **All in one place**: Built entirely with Python. One stop shop for all preprocessing steps including cropping, resizing, normalization, and model training.


