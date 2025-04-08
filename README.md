# isltranslationusingcnn

This project focuses on developing a deep learning-based pipeline for recognizing Indian Sign Language (ISL) from video clips. It processes raw videos, extracts relevant frames, applies preprocessing, and then utilizes both Vision Transformer (ViT) and Convolutional Neural Network (CNN) architectures to classify gestures accurately. The goal is to automate and enhance communication accessibility for the hearing and speech impaired using modern computer vision techniques.

1. Data Collection
Dataset is fetched via Zenodo API from the INCLUDE dataset for Indian Sign Language.
Videos are downloaded and stored in the directory:
/content/drive/MyDrive/INCLUDE_videos

2. Frame Extraction from Videos
OpenCV is used to extract one frame per second from each video.
Extracted frames are organized by video folder in:
/content/drive/MyDrive/frames

3. Image Preprocessing
All extracted frames are resized to 224x224 pixels.
Resized frames are saved in:
/content/drive/MyDrive/resized_frames

4. Feature Extraction with Vision Transformer (ViT)
A pre-trained Vision Transformer (google/vit-base-patch16-224-in21k) is used for feature extraction.
The CLS token from each frame is extracted as the feature vector.
Features are stored as .npy files in:
/content/drive/MyDrive/vit_features


5. CNN-Based Classification
 
a. Basic CNN
A basic convolutional structure with Conv2D, MaxPooling2D, Flatten, Dense, and Dropout layers.

b. CNN with Batch Normalization
The same as the basic CNN, but with BatchNormalization layers after every convolution for improved training stability and performance.

c. ResNet50
A pre-trained ResNet50 model is utilized as the feature extractor.
The last layers are tailored and trained for ISL classification.
The base model layers are frozen during early training.

d. LSTM on ViT Sequences
Sequences of ViT features (10 frames per video) are input into an LSTM for modeling temporal aspects.
The LSTM is followed by classification using fully connected layers.
The model identifies temporal dynamics of sign gestures.

e. Vision Transformer (ViT) with MLP Classifier
We use a pre-trained Vision Transformer (vit-base-patch16-224-in21k) to extract CLS token embeddings from an image.
These feature vectors are flattened and fed into a Multi-Layer Perceptron (MLP) with two dense layers, both of which include dropout regularization.
The MLP is trained to predict the features to be classified as one of the ISL sign categories.
