import time
from transformers import pipeline
import gradio as gr
import numpy as np
import librosa

transcriber_hindi = pipeline("automatic-speech-recognition", model="ai4bharat/indicwav2vec-hindi")
transcriber_bengali = pipeline("automatic-speech-recognition", model="ai4bharat/indicwav2vec_v1_bengali")
transcriber_odia = pipeline("automatic-speech-recognition", model="ai4bharat/indicwav2vec-odia")
transcriber_gujarati = pipeline("automatic-speech-recognition", model="ai4bharat/indicwav2vec_v1_gujarati")
# transcriber_telugu = pipeline("automatic-speech-recognition", model="ai4bharat/indicwav2vec_v1_telugu")
# transcriber_sinhala = pipeline("automatic-speech-recognition", model="ai4bharat/indicwav2vec_v1_sinhala")
# transcriber_tamil = pipeline("automatic-speech-recognition", model="ai4bharat/indicwav2vec_v1_tamil")
# transcriber_nepali = pipeline("automatic-speech-recognition", model="ai4bharat/indicwav2vec_v1_nepali")
# transcriber_marathi = pipeline("automatic-speech-recognition", model="ai4bharat/indicwav2vec_v1_marathi")

languages = ["hindi","bengali","odia","gujarati"]

def resample_to_16k(audio, orig_sr):
    y_resampled = librosa.resample(y=audio, orig_sr=orig_sr, target_sr=16000)
    return y_resampled

def transcribe(audio,lang="hindi"):
    sr,y = audio
    y = y.astype(np.float32)
    y/= np.max(np.abs(y))
    y_resampled = resample_to_16k(y,sr)
    if lang not in languages:
        return "No Model","So Stay tuned!"
    pipe= eval(f'transcriber_{lang}')
    start_time = time.time()
    trans = pipe(y_resampled)
    end_time = time.time()
    
    return trans["text"],(end_time-start_time)

demo = gr.Interface(
            transcribe,
            inputs=["microphone",gr.Radio(["hindi","bengali","odia","gujarati"],value="hindi")],
            # inputs=["microphone",gr.Radio(["hindi","bengali","odia","gujarati","telugu","sinhala","tamil","nepali","marathi"],value="hindi")],
            outputs=["text","text"],
            examples=[["./Samples/Hindi_1.mp3","hindi"],["./Samples/Hindi_2.mp3","hindi"],["./Samples/Hindi_3.mp3","hindi"],["./Samples/Hindi_4.mp3","hindi"],["./Samples/Hindi_5.mp3","hindi"],["./Samples/Tamil_2.mp3","hindi"],["./Samples/climate ex short.wav","hindi"],["./Samples/Gujarati_1.wav","gujarati"],["./Samples/Gujarati_2.wav","gujarati"],["./Samples/Bengali_1.wav","bengali"],["./Samples/Bengali_2.wav","bengali"]])
            # examples=[["./Samples/Hindi_1.mp3","hindi"],["./Samples/Hindi_2.mp3","hindi"],["./Samples/Tamil_1.mp3","tamil"],["./Samples/Tamil_2.mp3","hindi"],["./Samples/Nepal_1.mp3","nepali"],["./Samples/Nepal_2.mp3","nepali"],["./Samples/Marathi_1.mp3","marathi"],["./Samples/Marathi_2.mp3","marathi"],["./Samples/climate ex short.wav","hindi"]])
demo.launch()