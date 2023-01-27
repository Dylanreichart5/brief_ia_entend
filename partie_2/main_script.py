from joblib import load
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import sounddevice as sd
from features_functions import compute_features
# Get input signals that are different from the ones used during training


#create a sinus sound
random_number = random.randint(0, 1)
fe = 44100
amp = (random.randint(1, 7))/10
duree = (random.randint(1, 6))/2 # nombre de seconde
if random_number == 0 :
    f0 = random.randint(20, 20000) #la fréquence audible est comprise entre 1 et 20000 hz
    t = np.arange(0, duree, 1/fe)
    type_son = random.randint(1,2)
    sound = amp*np.sin(2*np.pi*f0*t)
    print("le son est un son sinusoidale")
else : 
    sound = amp*np.random.randn(int(duree*fe))
    print("le son est un bruit blanc ")
#get input singal

sig_sq = sound**2
sig_t = sound / np.sqrt(sig_sq.sum())
sig_f = np.absolute(np.fft.fft(sig_t))
sig_c = np.absolute(np.fft.fft(sig_f))

# Compute the features on these signals
features_list = []
N_feat, features_list = compute_features(sig_t, sig_f[:sig_t.shape[0]//2], sig_c[:sig_t.shape[0]//2])
features_vector = np.array(features_list)[np.newaxis,:]

FEATURES = features_vector

# Load the scaler and SVM model to test the class of your source
scaler = load("SCALER")
model = load("SVM_MODEL")


# Get the class of your source
prediction = model.predict(FEATURES)


# Analyze the results
    
if prediction == 0:
    print("l'algo prédit un son sinusoidale")
else:
    print("l'algo prédit un bruit blanc")