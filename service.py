import flask
from flask import request, jsonify, abort, send_from_directory
from scipy.io import wavfile
import json
from json import JSONEncoder
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import threading
import time
import os
import subprocess

app = flask.Flask(__name__)
app.config["DEBUG"] = True

dataPath = "data"
url = "test.wav"
synthesizer = ""

def setup():
    global synthesizer
    encoder_weights = Path("encoder/saved_models/pretrained.pt")
    vocoder_weights = Path("vocoder/saved_models/pretrained/pretrained.pt")
    syn_dir = Path("synthesizer/saved_models/logs-pretrained/taco_pretrained")
    encoder.load_model(encoder_weights)
    synthesizer = Synthesizer(syn_dir)
    vocoder.load_model(vocoder_weights)

def run_voiceCloning(filename):   
    in_fpath = dataPath + "/" + filename
    textPath = dataPath + "/" + filename + ".txt"
    
    subprocess.call(['ffmpeg', '-i', in_fpath + '.mp3',
                   in_fpath + '.wav'])
    time.sleep(5)
    #reprocessed_wav = encoder.preprocess_wav(in_fpath)
    original_wav, sampling_rate = librosa.load(Path(in_fpath + '.wav'))
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    embed = encoder.embed_utterance(preprocessed_wav)
    
    textFile = open(textPath)
    text = textFile.read().replace("\n", " ")
    textFile.close()
    
    specs = synthesizer.synthesize_spectrograms([text], [embed])
  
    generated_wav = vocoder.infer_waveform(specs[0])
    
    return np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

def clean(delay, path):
    time.sleep(delay)
    print("Cleaning [START]: " + path)
    
    os.remove(dataPath + "/" + path + ".txt")
    os.remove(dataPath + "/" + path + ".mp3")
    os.remove(dataPath + "/" + path + ".wav")
    os.remove(dataPath + "/" + path + "_out.mp3")
    os.remove(dataPath + "/" + path + "_out.wav")
    
    print("Cleaning [END|SUCCESS]: " + path)

@app.route('/audio/create', methods=['POST'])
def post_file():
    files = flask.request.files.getlist("file")
    receivedFiles = ""
    for file in files:
        receivedFiles = receivedFiles + file.filename + "  "
    print("Received :", receivedFiles)
    
    sharedFileName = files[0].filename.split('.')[0]
    
    if(sharedFileName != files[1].filename.split('.')[0]):
        print("Files do not share the same name")
        abort(400, "All files must have the same name" + sharedFileName)
    
    for file in files:
        if "/" in file.filename:
            abort(400, "Subdirectories are not allowed")
        else:
            file.save(open(os.path.join(dataPath, file.filename), "wb"))
             
    wavfile.write(dataPath + "/" + sharedFileName + "_out.wav", synthesizer.sample_rate, run_voiceCloning(sharedFileName))
    
    in_fpath = dataPath + "/" + sharedFileName + "_out"
    subprocess.call(['ffmpeg', '-i', in_fpath + '.wav',
                   in_fpath + '.mp3'])
    
    clearer = threading.Thread(target=clean, args=(10, sharedFileName))
    clearer.start()
    
    return send_from_directory(dataPath, sharedFileName + "_out.mp3", as_attachment=True)
    	
@app.route("/ip", methods=["GET"])
def getIp():
    return jsonify({'ip': request.remote_addr}), 200
    	
@app.route('/')
def home():
    homepage = "<div>"
	
    homepage = homepage + "<p>Welcome to VoiceCloning Web</p>"
    homepage = homepage + "<p>Created by Vlad-Alexandru Iacob</p>"
    homepage = homepage + "<p><hr></p>"
    homepage = homepage + "<p>This service creates a audio file which contains a spoken text using a given voice</p>"
    homepage = homepage + "<p>To make such a file, post an .mp3 or .wav and a .txt file to /audio/create, and make sure they share filenames.</p>"
    homepage = homepage + "<p>You will receive the created file as an attachment in a couple of seconds later.</p>"
    homepage = homepage + "<p>All data is deleted when the creation process is finished, this is a stateless service.</p>"
	
    return homepage + "</div>";
    	
if __name__ == "__main__":
    setup()
    app.run(host='192.168.0.112')
