import numpy as np
import joblib

def load_features(file_path):
    # This function loads features from a specified .pkl file
    features, labels = joblib.load(file_path)
    return features, labels

def combine_features(features1, features2):
    # This function combines two feature arrays by concatenation
    # Ensure that both feature sets have the same number of samples
    assert features1.shape[0] == features2.shape[0], "The number of samples in the feature sets do not match."
    combined_features = np.hstack((features1, features2))
    return combined_features

def save_combined_features(features, labels, file_path):
    # This function saves the combined features to a .pkl file
    joblib.dump((features, labels), file_path)

def main():
    # Paths to the feature files
    train_edge_features_path = 'train_edge_hough_features.pkl'
    test_edge_features_path = 'test_edge_hough_features.pkl'
    train_pca_features_path = 'train_pca_features_top100.pkl'
    test_pca_features_path = 'test_pca_features_top100.pkl'
    
    # Load features
    train_edge_features, train_edge_labels = load_features(train_edge_features_path)
    train_pca_features, train_pca_labels = load_features(train_pca_features_path)

    # Load features
    test_edge_features, test_edge_labels = load_features(test_edge_features_path)
    test_pca_features, test_pca_labels = load_features(test_pca_features_path)
    
    # Ensure that the labels match
    assert np.array_equal(train_edge_labels, train_pca_labels), "Labels from edge features and PCA features do not match."
    
    assert np.array_equal(test_edge_labels, test_pca_labels), "Labels from edge features and PCA features do not match."

    # Combine features
    train_combined_features = combine_features(train_edge_features, train_pca_features)
    test_combined_features = combine_features(test_edge_features, test_pca_features)
    
    # Save combined features
    save_combined_features(train_combined_features, train_edge_labels, 'train_combined_features.pkl')
    print("Combined features have been saved to 'train_combined_features.pkl'.")
    
    save_combined_features(test_combined_features, test_edge_labels, 'test_combined_features.pkl')
    print("Combined features have been saved to 'test_combined_features.pkl'.")

if __name__ == '__main__':
    main()