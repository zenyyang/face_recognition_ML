# Face Recognition ML
A Final project for Machine Learning program, to develop a Face Recognition model.
This GitHub repository contains code for face recognition using Support Vector Machines (SVM) and Principal Component Analysis (PCA). The code is designed to work with the Labeled Faces in the Wild (LFW) dataset. 

### Requirements
Make sure you have the following Python libraries installed:
```python
pip install numpy pandas scikit-learn opencv-python scikit-image matplotlib seaborn colorama
```

### Getting Started
1. Clone this repo:
```python
git clone https://github.com/your-username/your-repo.git
cd your-repo
```
2.You will need to download the LFW dataset from its website or this link [LFW](http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz.). Place the folder of images in the root folder of the project.

### Making Predictions
```python
import pickle
from skimage.io import imread
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the trained model and PCA object
with open("face_detection.pkl", 'rb') as file:
    svm = pickle.load(file)

with open("pca.pkl", 'rb') as file:
    pca = pickle.load(file)

# Load a new image for prediction
image_path = "path/to/your/image.jpg"
image = imread(image_path, as_gray=True)

# Normalize and reshape the image
scaler = StandardScaler()
image_std = scaler.fit_transform(image.flatten().reshape(1, -1))

# Apply PCA
image_pca = pca.transform(image_std)

# Make predictions
prediction = svm.predict(image_pca)

print(f"The predicted label for {image_path} is:", prediction)

```
Replace "path/to/your/image.jpg" with the path to your image.

### Learn More
1. [Labeled Faces in the Wild (LFW) dataset](https://vis-www.cs.umass.edu/lfw/)
2. [OpenCV](https://opencv.org/)
3. [Scikit-learn](https://scikit-learn.org/stable/)
