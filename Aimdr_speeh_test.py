import whisper
import librosa
import torch
import whisper.audio as wa
from sklearn.cluster import KMeans
import numpy as np

#weak for rearch purposes switch to pyannote

# Load Whisper model  
model = whisper.load_model("base", device="cpu")

# You can use: tiny, base, small, medium, large

audio_path= "amicorpus/ES2008a/audio/ES2008a.Mix-Headset.wav"

y,sr =librosa.load(audio_path,sr=16000)
# Transcribe AMI meeting audio
result = model.transcribe(
    audio_path,word_timestamps=True
)

segments=result['segments']

#extract mfcc features per segment
def extract_mfcc(y,sr,start,end):
    sample_start=int(start *sr) 

    sample_end=int(end *sr)

    segment_audio=y[sample_start:sample_end]

    mfcc=librosa.feature.mfcc(y=segment_audio,sr=sr,n_mfcc=13)

    return np.mean(mfcc,axis=1)

#Create feature matrix

features=[]
valid_segments=[]

for seg in segments:
    if seg["end"]-seg['start']>1.0:
        mfcc_feat=extract_mfcc(
            y,sr,seg['start'],seg["end"]
        )

        features.append(mfcc_feat)
        valid_segments.append(seg)

features_array=np.array(features)

kmeans=KMeans(n_clusters=4,random_state=42)

speaker_labels=kmeans.fit_predict(features_array)

# Save transcript
'''with open("ES2008a_transcript2.txt", "w") as f:
    f.write(result["text"])

print("Transcription completed.")

'''

# Load audio and convert
'''audio = wa.load_audio(audio_path)
audio = wa.pad_or_trim(audio)

# Convert to log-Mel spectrogram
mel_spectrogram = wa.log_mel_spectrogram(audio)

print("Mel Spectrogram shape:", mel_spectrogram.shape)'''

for i,seg in enumerate(valid_segments):
    print(f"Spaker{speaker_labels[i]} "
    f"[{seg['start']:.2f}-{seg['end']:.2f}]: "
    f"{seg['text']}")



