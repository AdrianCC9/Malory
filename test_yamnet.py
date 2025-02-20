import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa

# Load YAMNet model
yamnet_model = hub.load('/Users/adrian/models/YAMNet')

# Load audio file
audio_file = '/Users/adrian/models/YAMNet/sample.wav'
waveform, sample_rate = librosa.load(audio_file, sr=16000)

# Run YAMNet
scores, embeddings, spectrogram = yamnet_model(waveform)

# Get top 5 predictions
class_map_path = '~/models/YAMNet/assets/yamnet_class_map.csv'
class_names = [line.split(',')[2].strip() for line in open(class_map_path).readlines()[1:]]
top5_i = np.argsort(scores.numpy()[0])[-5:][::-1]

# Print predictions
print("\nTop 5 Predictions:")
for i in top5_i:
    print(f"{class_names[i]} ({scores.numpy()[0][i]:.3f})")
