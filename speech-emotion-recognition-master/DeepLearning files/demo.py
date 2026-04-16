from common import extract_data
from sklearn.neural_network import MLPClassifier

# data load

x_train, x_test, y_train, y_test, _ = extract_data(flatten=True)

# model train

model = MLPClassifier(alpha=0.01, batch_size=256, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
model.fit(x_train, y_train)

# prediction on one sample

sample = x_test[0].reshape(1, -1)
prediction = model.predict(sample)

# emotion labels

emotion_map = {
0: "Neutral",
1: "Angry",
2: "Happy",
3: "Sad"
}

print("Predicted Emotion:", emotion_map[prediction[0]])
