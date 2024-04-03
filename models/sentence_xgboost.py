import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from pyPPG.datahandling import load_data
import pyPPG.preproc as PP
from scipy.fft import  fft, rfft, rfftfreq
import numpy as np
from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

sampling_rate = 64

def processPPG(file_path):
    signal = load_data(file_path,fs=sampling_rate)
    signal.filtering = True # whether or not to filter the PPG signal
    signal.fL=0.5000001 # Lower cutoff frequency (Hz)
    signal.fH=30 # Upper cutoff frequency (Hz)
    signal.order=2 # Filter order
    # signal.sm_wins={'ppg':8,'vpg':8,'apg':8,'jpg':8} # smoothing windows in millisecond for the PPG, PPG', PPG", and PPG'"
    prep = PP.Preprocess(fL=signal.fL, fH=signal.fH, order=signal.order, sm_wins=signal.sm_wins)
    signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(s=signal)
    ppg_fft = np.fft.rfft(signal.ppg)

    magnitude_spectrum = np.abs(ppg_fft)

    # freqs = np.fft.rfftfreq(len(signal), 1/sampling_rate) 
    psd = np.abs(ppg_fft) ** 2

    cumulative_energy = np.cumsum(psd)
    total_energy = cumulative_energy[-1]

    normalized_psd = psd / total_energy
    spectral_entropy = -np.sum(normalized_psd * np.log2(normalized_psd + 1e-10))

    spectral_flatness = np.exp(np.mean(np.log(magnitude_spectrum + 1e-10))) / np.mean(magnitude_spectrum)
    return [spectral_flatness, spectral_entropy] 



def getSentenceVector(file_path):
    f = open(file_path,'r')
    description = f.readline()
    embeddings = sentence_model.encode(description)
    
    # print(embeddings)
    # exit()
    return list(embeddings)

def loadPPG_Sentence(dirs):
    X = []
    Y = []
    Actions = os.listdir(dirs)
    for i in range(len(Actions)):
        tmp = getSentenceVector(os.path.join(dirs,Actions[i],"description.txt"))
        for s in os.listdir(os.path.join(dirs,Actions[i],"PPG")):
            X.append(processPPG(os.path.join(dirs,Actions[i],"PPG",s))+tmp)
            # X.append(tmp)
            Y.append(i)

    return X,Y



X, y = loadPPG_Sentence("datasets")

# iris = load_iris()
# X = iris.data
# y = iris.target
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100.0}%")
