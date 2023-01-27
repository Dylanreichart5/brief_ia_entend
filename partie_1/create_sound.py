# Import des packages nécessaires
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import sounddevice as sd


bdd_sound = []
def sound_sinus():
    fe = 44100
    for i in range (10):
        duree = (random.randint(1, 6))/2 # nombre de seconde
        f0 = random.randint(20, 20000) #la fréquence audible est comprise entre 1 et 20000 hz
        amp = (random.randint(1, 7))/10
        t = np.arange(0, duree, 1/fe)
        type_son = random.randint(1,2)
        sinus = amp*np.sin(2*np.pi*f0*t)
        fichier_sound = open("Data/{}".format('fichier_sound'+str(i)),"wb")
        pickle.dump(sinus, fichier_sound)
        
    for i in range (10):
        bb = amp*np.random.randn(int(duree*fe))
        fichier_sound = open("Data/{}".format("fichier_sound"+str(10+i)),"wb") 
        pickle.dump(bb, fichier_sound) 
sound_sinus()

