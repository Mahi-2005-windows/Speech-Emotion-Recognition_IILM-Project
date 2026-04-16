#"""
#This example demonstrates how to use `NN` model ( any ML model in general) from
#`speechemotionrecognition` package
#"""
#from common import extract_data
#from speechemotionrecognition.mlmodel import NN
#from speechemotionrecognition.utilities import get_feature_vector_from_mfcc


#def ml_example():
 #   to_flatten = True
  #  x_train, x_test, y_train, y_test, _ = extract_data(flatten=to_flatten)
   # model = NN()
    #print('Starting', model.name)
    #model.train(x_train, y_train)
    #model.evaluate(x_test, y_test)
    #filename = '../dataset/Sad/09b03Ta.wav'
    #print('prediction', model.predict_one(
     #   get_feature_vector_from_mfcc(filename, flatten=to_flatten)),
      #    'Actual 3')


#if __name__ == "__main__":
 #   ml_example()


#NEW UPDATED CODE

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from speechemotionrecognition.mlmodel import SVM
from speechemotionrecognition.utilities import get_data, extract_features


def ml_example():
    print("Loading data...")

    # Load dataset
    X, y = get_data("dataset")

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔥 Best Model (SVM)
    model = SVM()

    print('Training', model.name)
    model.train(x_train, y_train)

    # Predictions
    y_pred = model.model.predict(x_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    # Detailed report
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # 🔥 Single prediction test
    filename = 'dataset/Sad/09b03Ta.wav'
    features = extract_features(filename)

    print('Prediction:', model.predict_one(features), 'Actual: 3')


if __name__ == "__main__":
    ml_example()