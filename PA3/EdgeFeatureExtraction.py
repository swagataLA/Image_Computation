import numpy as np
import cv2
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import joblib

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

def getLineIntersection(line1, line2):
    #calculating intersection from HoughLines returned lines
    pointFromOrigin1, angle1 = line1[0]
    pointFromOrigin2, angle2 = line2[0]

    a1, b1 = np.cos(angle1), np.sin(angle1)
    a2, b2 = np.cos(angle2), np.sin(angle2)

    determinant = a1 * b2 - a2 * b1

    if abs(determinant) < 1e-10:
        return None  # Lines are parallel
    
    xIntersection = (b2 * pointFromOrigin1 - b1 * pointFromOrigin2) / determinant
    yIntersection = (a1 * pointFromOrigin2 - a2 * pointFromOrigin1) / determinant
    
    return int(xIntersection), int(yIntersection)

def getEdgeFeatures(imageArray):
    edgeFeatures = []
    for image in imageArray:
        gray_image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image
        edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=50)
        
        if lines is not None:
            intersections = []
            numLines = len(lines)
            for i in range(numLines):
                for j in range(i + 1, numLines):
                    intersect = getLineIntersection(lines[i], lines[j])
                    if intersect:
                        intersections.append(intersect)
            
            
            featureImage = np.zeros_like(edges)
            for (x, y) in intersections:
                if 0 <= x < featureImage.shape[1] and 0 <= y < featureImage.shape[0]:
                    #marking the feature on the nparray
                    featureImage[y, x] = 255

            edgeFeatures.append(featureImage.flatten())
        else:
            edgeFeatures.append(edges.flatten())

    return np.array(edgeFeatures)


def main():
    (trainImages, trainLabels), (testImages, testLabels) = load_data('C:/Users/swaga/Documents/CS510/PA1/cs510tutorials (1)/cs510tutorials/PA3/dataset')
    
    # Normalizing values
    trainImages = trainImages / 255.0
    testImages = testImages / 255.0
    
    # Extract edge features
    trainEdgeFeatures = getEdgeFeatures(trainImages)
    testEdgeFeatures = getEdgeFeatures(testImages)
    
    # Save extracted features and labels
    joblib.dump((trainEdgeFeatures, trainLabels), 'train_edge_hough_features.pkl')
    joblib.dump((testEdgeFeatures, testLabels), 'test_edge_hough_features.pkl')

if __name__ == '__main__':
    main()
