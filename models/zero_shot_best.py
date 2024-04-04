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
import torch.nn as nn
import torch.optim as optim
import torch

seed_value=42
torch.manual_seed(seed_value)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
Known_Actions=["sitting","walking"]
Unknown_Actions=["stairs"]
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
    return [np.mean(magnitude_spectrum),spectral_flatness, spectral_entropy] 



def getSentenceVector(file_path):
    f = open(file_path,'r')
    description = f.readline()
    embeddings = sentence_model.encode(description)
    
    # print(embeddings)
    # exit()
    return list(embeddings)

def loadPPG_Sentence(dirs,Actions):
    txt_features=[]
    ppg_features=[]
    for i in range(len(Actions)):
        tmp = getSentenceVector(os.path.join(dirs,Actions[i],"description.txt"))
        for s in os.listdir(os.path.join(dirs,Actions[i],"PPG")):
            txt_features.append(tmp)
            ppg_features.append(processPPG(os.path.join(dirs,Actions[i],"PPG",s)))
    return txt_features, ppg_features

class ContrastiveModel(nn.Module):
    def __init__(self, sentence_feature_dim, PPG_feature_dim, embed_dim):
        super().__init__()
        self.txt_linear = nn.Linear(sentence_feature_dim, embed_dim)
        self.ppg_linear = nn.Linear(PPG_feature_dim, embed_dim)
        
        
    
    def forward(self, txt_feature, ppg_feature):
        txt_embed = self.txt_linear(torch.tensor(txt_feature, dtype=torch.float32))
        ppg_embed = self.ppg_linear(torch.tensor(ppg_feature, dtype=torch.float32))
        
        
        return txt_embed,ppg_embed 

def contrastive_loss(Map, margin=1.0):
    cos = nn.CosineSimilarity(dim=1)
    loss = 0
    positive_loss=0
    negative_loss=0

    positives=[]
    negatives=[]
    for act in Map:
        positive_similarity = cos(Map[act][0], Map[act][1])
        positives.append((1 - positive_similarity).clamp(min=0.0))
    
    positive_loss = torch.mean(torch.stack(positives))

    for act1 in Map:
        for act2 in Map:
            if act1==act2: continue
            negatives.append(cos(Map[act1][0], Map[act2][1]).clamp(min=0.0))
            negatives.append(cos(Map[act2][0], Map[act1][1]).clamp(min=0.0))

    negative_loss = torch.mean(torch.stack(negatives))
    loss = positive_loss.mean() + negative_loss.mean()
    return loss.clamp(min=0.0)


model = ContrastiveModel(sentence_feature_dim=384,PPG_feature_dim=3, embed_dim=128)
optimizer = optim.Adam(model.parameters(), lr=0.003)
epochs = 50

Map = {}
for epoch in range(epochs):
    optimizer.zero_grad()
    loss=0
    for act in Known_Actions:
        txt_features, ppg_features = loadPPG_Sentence("datasets",[act])
        txt_embed, ppg_embed = model(txt_features, ppg_features)
        Map[act] = [txt_embed, ppg_embed]
    loss = contrastive_loss(Map).mean()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    # exit()

model.eval() 
with torch.no_grad():
    cos_sim = nn.CosineSimilarity(dim=1)

    # Correct Unknown
    txt_features, ppg_features = loadPPG_Sentence("datasets",Unknown_Actions)
    unknown_txt_embed, unknown_ppg_embed = model(txt_features, ppg_features)
    similarity = cos_sim(unknown_txt_embed,unknown_ppg_embed)
    print("%s's posbiility: %f"%(Unknown_Actions[0],similarity.mean()))

    # exit()
    for act in Known_Actions:
        txt_features, ppg_features = loadPPG_Sentence("datasets",[act])
        txt_embed,_ = model(txt_features, ppg_features)
        similarity = cos_sim(txt_embed,unknown_ppg_embed)
        print("%s's posbiility: %f"%(act,similarity.mean()))



