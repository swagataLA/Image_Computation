import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm  import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Lasso
# from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set() 

def getData(clf,testPCAFeatures, testLabels):
    predictions = clf.predict(testPCAFeatures)
    
     # Evaluate the classifier
    accuracy = accuracy_score(testLabels, predictions)
    precision = precision_score(testLabels, predictions, average='weighted')
    recall = recall_score(testLabels, predictions, average='weighted')

    # Print the metrics
    print(f'Test Accuracy: {accuracy * 100.0:.2f}%')
    print(f'Test Precision: {precision * 100.0:.2f}%')
    print(f'Test Recall: {recall * 100:.2f}%')


def main():
    if len(sys.argv) > 1:
        print("Arguments received:")
        new_var = sys.argv
        for arg in new_var[1:]:
            print(arg)
    else:
        print("No arguments provided.")

    model = sys.argv[1]

    trainingData ='train_combined_features.pkl'
    testData= 'test_combined_features.pkl'

    trainPCAFeatures, train_labels = joblib.load(trainingData)
    testPCAFeatures, testLabels = joblib.load(testData)
    
    if(model == "1"):
        dept = None
        if len(sys.argv) > 3:
            dept = int(sys.argv[3])
        clf = RandomForestClassifier(n_estimators=int(sys.argv[2]),max_depth=dept ,random_state=42)
        # print
    if(model == "2"):
        clf = SVC(C=float(sys.argv[2]), kernel=sys.argv[3], random_state=42)
    if(model == "3"):
        clf = LogisticRegression(C=float(sys.argv[2]), solver=sys.argv[3], random_state=42)
    if(model == "4"):
            clf = KNeighborsClassifier(n_neighbors=int(sys.argv[2]),weights='uniform')
    if(model == "5"):
        clf = DecisionTreeClassifier(max_depth=3, criterion=sys.argv[2], random_state=42)
    if(model == "6"):
        clf = GradientBoostingClassifier(n_estimators=int(sys.argv[2]), learning_rate=0.1, max_depth=3, random_state=42)
    if(model == "7"):
        clf = AdaBoostClassifier(
            # base_estimator= DecisionTreeClassifier(max_depth=1),
            n_estimators=int(sys.argv[2]),
            learning_rate=1.0,
            random_state=42
        )
    if(model == "8"):
        clf = GaussianNB(var_smoothing=float(sys.argv[2]))
    if(model == "9"):
        clf = Lasso(alpha=float(sys.argv[2]), random_state=42)
    if(model == "10"):
        clf =  SGDClassifier(loss=sys.argv[2], max_iter=1000, tol=1e-3, random_state=42)
    
#         clf = DecisionTreeClassifier(max_depth=3, criterion=sys.argv[2], random_state=42)
#    if(model == "8"):
#         clf = DecisionTreeClassifier(max_depth=3, criterion=sys.argv[2], random_state=42)
#    if(model == "9"):
#         clf = DecisionTreeClassifier(max_depth=3, criterion=sys.argv[2], random_state=42)
#     if(model == "10"):
#         clf = DecisionTreeClassifier(max_depth=3, criterion=sys.argv[2], random_state=42)
   
    
    
    
    clf.fit(trainPCAFeatures, train_labels)
    getData(clf,testPCAFeatures, testLabels)
    


if __name__ == '__main__':
    main()