from sklearn.decomposition import PCA
# from tensorflow.keras.datasets import mnist
import joblib
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split

# Function to load data and resize images
def load_data(data_dir, target_size=(28, 28)):
    images = []
    labels = []
    # Traverse all subdirectories representing each label
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                if img_name.endswith('.png'):
                    img_path = os.path.join(label_path, img_name)
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    # Resize image to the desired target size
                    img = img.resize(target_size)
                    img_array = np.array(img)
                    images.append(img_array)
                    labels.append(int(label))

    # Convert lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    return (x_train, y_train), (x_test, y_test)


def main():
    
    # Load MNIST dataset
    (trainImages, trainLabels), (testImages, testLabels) = load_data('C:/Users/swaga/Documents/CS510/PA1/cs510tutorials (1)/cs510tutorials/PA3/dataset')
    
    # Flatten images
    trainImages = trainImages.reshape((trainImages.shape[0], -1))
    testImages = testImages.reshape((testImages.shape[0], -1))
    
    # Normalize
    trainImages = trainImages / 255.0
    testImages = testImages / 255.0

    pca = PCA(n_components=100)

    # Apply PCA on the training set to get top 100 features
    trainPCAFeatures = pca.fit_transform(trainImages)
    
    # Transform the test set using the same PCA
    testPCAFeatures = pca.transform(testImages)
    
    # Save PCA features and labels
    joblib.dump((trainPCAFeatures, trainLabels), 'train_pca_features_top100.pkl')
    joblib.dump((testPCAFeatures, testLabels), 'test_pca_features_top100.pkl')

if __name__ == '__main__':
    main()
