import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set() 

def evaluateRandomForest(trainingData, testData):
    trainPCAFeatures, train_labels = joblib.load(trainingData)
    testPCAFeatures, testLabels = joblib.load(testData)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    clf.fit(trainPCAFeatures, train_labels)
    
    predictions = clf.predict(testPCAFeatures)
    probabilities = clf.predict_proba(testPCAFeatures)
    
     # Evaluate the classifier
    accuracy = accuracy_score(testLabels, predictions)
    precision = precision_score(testLabels, predictions, average='weighted')
    recall = recall_score(testLabels, predictions, average='weighted')
    f1 = f1_score(testLabels, predictions, average='weighted')
    auc = roc_auc_score(testLabels, probabilities, multi_class='ovr', average='weighted')
    cm = confusion_matrix(testLabels, predictions)

    # Print the metrics
    print(f'Test Accuracy: {accuracy * 100.0:.2f}%')
    print(f'Test Precision: {precision * 100.0:.2f}%')
    print(f'Test Recall: {recall * 100:.2f}%')
    print(f'Test F1 Score: {f1* 100:.2f}%')
    print(f'Test AUC: {auc * 100:.2f}%')
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def main():

    mode = ''
    if len(sys.argv) > 1:
        print("Arguments received:")
        for arg in sys.argv[1:]:
            mode = arg
    else:
        print("No arguments provided.")

    trainingData = None
    testData = None
    
    if(mode == "-edges"):
        trainingData ='train_edge_hough_features.pkl'
        testData= 'test_edge_hough_features.pkl'
    elif(mode == "-pca"):
        trainingData ='train_pca_features_top100.pkl'
        testData= 'test_pca_features_top100.pkl'
    elif(mode == "-both"):
        trainingData ='train_combined_features.pkl'
        testData= 'test_combined_features.pkl'
      
    evaluateRandomForest(trainingData, testData)

if __name__ == '__main__':
    main()