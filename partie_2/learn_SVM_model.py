import random
import numpy as np
from sklearn import preprocessing
from sklearn import svm
import pickle
from joblib import dump
from features_functions import compute_features
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

learningFeatures = np.empty((20,71))
learningLabels = np.zeros((20))
for i in range(20):
    # Get an input signal
    file = open("Data/{}".format("fichier_sound"+str(i)), 'rb')
    input_sig = pickle.load(file)

    # Compute the signal in three domains
    sig_sq = input_sig**2
    sig_t = input_sig / np.sqrt(sig_sq.sum())
    sig_f = np.absolute(np.fft.fft(sig_t))
    sig_c = np.absolute(np.fft.fft(sig_f))
    # Compute the features and store them
    features_list = []
    N_feat, features_list = compute_features(sig_t, sig_f[:sig_t.shape[0]//2], sig_c[:sig_t.shape[0]//2])
    features_vector = np.array(features_list)[np.newaxis,:]

    # Store the obtained features in a np.arrays
    if i < 10:
        learningFeatures[i] = features_vector # 2D np.array with features_vector in it, for each signal
    else:
        learningFeatures[i] = features_vector 


    # Store the labels
    if i < 10:
        learningLabels[i] = 0 # Les sinus
    else:
        learningLabels[i] = 1 # Les bruits blancs

X_train, X_test, y_train, y_test = train_test_split(learningFeatures, learningLabels, test_size=0.3)

# Encode the class names
labelEncoder = preprocessing.LabelEncoder().fit(y_train)
learningLabelsStd = labelEncoder.transform(y_train)

# Learn the model
model = svm.SVC(C=10, kernel='linear', class_weight=None, probability=False)
scaler = preprocessing.StandardScaler(with_mean=True).fit(X_train)

learningFeatures_scaled = scaler.transform(X_train)
model.fit(learningFeatures_scaled, learningLabelsStd)



# Make predictions on the test data
y_pred = model.predict(X_test)

# # Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# Export the scaler and model on disk
dump(scaler, "SCALER")
dump(model, "SVM_MODEL")

