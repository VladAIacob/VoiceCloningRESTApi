import flask
from flask import request, jsonify, abort, send_from_directory
from scipy.io import wavfile
from encoder import inference as encoder
from pathlib import Path
import numpy as np
import subprocess
import threading
import librosa
import torch
import time
import json
import sys
import os

sys.path.insert(0, "flowtron")
from flowtron import Flowtron
from torch.utils.data import DataLoader
from data import Data

sys.path.insert(0, "flowtron/tacotron2")
sys.path.insert(0, "flowtron/tacotron2/waveglow")
from glow import WaveGlow


app = flask.Flask(__name__)
app.config["DEBUG"] = True

#Data paths
examplePath = "audio/examples"
dataPath = "audio/data"

#Create example embeds for an audio file
def exampleEmbed(filename):
    in_fpath = examplePath + "/" + filename
    original_wav, sampling_rate = librosa.load(Path(in_fpath))
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    #getting the embeds from the encoder
    return encoder.embed_utterance(preprocessed_wav)

#Create the example embeds
def examplesSetup():
    global barackobama
    global gordonRamsay
    global stephenHawking
    
    barackobama = exampleEmbed("barackobama.wav")
    gordonRamsay = exampleEmbed("gordonRamsay.wav")
    stephenHawking = exampleEmbed("stephenHawking.wav")

#Initializes the components with pretrained weights
def setup():
    # Parse configs.  Globals nicer in this case
    with open("flowtron/infer.json") as f:
        data = f.read()

    global config
    config = json.loads(data)

    global data_config
    data_config = config["data_config"]
    global model_config
    model_config = config["model_config"]

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    global flowtron
    global waveglow
    global trainset
    
    encoder_weights = Path("encoder/saved_models/pretrained.pt")
    encoder.load_model(encoder_weights)

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    #Load waveglow
    waveglow = torch.load("flowtron/tacotron2/waveglow/saved_models/waveglow_256channels_universal_v5.pt")['model'].cuda().eval()
    waveglow.cuda().half()
    for k in waveglow.convinv:
        k.float()
    waveglow.eval()
    
    #Load flowtron
    flowtron = Flowtron(**model_config).cuda()
    state_dict = torch.load("flowtron/saved_models/pretrained.pt", map_location='cpu')['model'].state_dict()
    flowtron.load_state_dict(state_dict)
    flowtron.eval()

    ignore_keys = ['training_files', 'validation_files']
    trainset = Data(
        data_config['training_files'],
        **dict((k, v) for k, v in data_config.items() if k not in ignore_keys))

#Generate audio from embeds
def audioFromEmbeds(filename, embed):
    textPath = dataPath + "/" + filename + ".txt"
    #reading the text file and prepare the text string for the synthesizer
    textFile = open(textPath)
    text = textFile.read().replace("\n", " ")
    textFile.close()
    
    text = trainset.get_text(text).cuda()
    embeds = trainset.get_embeds(embed).cuda()
    text = text[None]
    embeds = embeds[None]
    
    with torch.no_grad():
        residual = torch.cuda.FloatTensor(1, 80, 400).normal_() * 0.5
        mels, attentions = flowtron.infer(
            residual, embeds, text, gate_threshold=0.5)

    with torch.no_grad():
        audio = waveglow.infer(mels.half(), sigma=0.8).float()

    audio = audio.cpu().numpy()[0]
    # normalize audio for now
    audio = audio / np.abs(audio).max()
    
    return audio

#Run the application on the 2 files sharing the name 'filename'
def run_voiceCloning(filename):   
    in_fpath = dataPath + "/" + filename
    
    #transforming mp3 into wav
    subprocess.call(['ffmpeg', '-i', in_fpath + '.mp3',
                   in_fpath + '.wav'])
    time.sleep(5)
    #running the encoder on the audio input
    original_wav, sampling_rate = librosa.load(Path(in_fpath + '.wav'))
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    #getting the embeds from the encoder
    embed = encoder.embed_utterance(preprocessed_wav)
    
    return audioFromEmbeds(filename, [embed])

#Delete all uploaded files when using examples
def cleanExample(delay, path):
    time.sleep(delay)
    print("Cleaning [START|Example]: " + path)
    
    os.remove(dataPath + "/" + path + ".txt")
    os.remove(dataPath + "/" + path + "_out.mp3")
    os.remove(dataPath + "/" + path + "_out.wav")
    
    print("Cleaning [END|SUCCESS>Example]: " + path)

#Delete all uploaded files to the service and the created auxiliary files by the service
def clean(delay, path):
    time.sleep(delay)
    print("Cleaning [START]: " + path)
    
    os.remove(dataPath + "/" + path + ".txt")
    os.remove(dataPath + "/" + path + ".mp3")
    os.remove(dataPath + "/" + path + ".wav")
    os.remove(dataPath + "/" + path + "_out.mp3")
    os.remove(dataPath + "/" + path + "_out.wav")
    
    print("Cleaning [END|SUCCESS]: " + path)
    
#Example list page
@app.route('/audio/example', methods=['GET'])
def listExamples():
    list = "<div><hr>"
    list = list + "<p>Barack Obama example    : /audio/example/obama</p><hr>"
    list = list + "<p>Gordon Ramsay example   : /audio/example/ramsay</p><hr>"
    list = list + "<p>Stephen Hawking example : /audio/example/hawking</p><hr>"
    return list + "</div>"

#Barack Obama Example
@app.route('/audio/example/obama', methods=['POST'])
def obama():
    return examplePage(barackobama)

#Gordon Ramsay Example
@app.route('/audio/example/ramsay', methods=['POST'])
def ramsay():
    return examplePage(gordonRamsay)
    
#Stephen Hawking Example
@app.route('/audio/example/hawking', methods=['POST'])
def hawking():
    return examplePage(stephenHawking)

#Template for example post request page
def examplePage(embed):
    files = flask.request.files.getlist("file")
    receivedFiles = ""
    #debuging: print the files received
    for file in files:
        receivedFiles = receivedFiles + file.filename + "  "
    print("Received :", receivedFiles)
    
    #check for receiving only 1 file
    if len(files) != 1:
        print("More or less files received")
        abort(400, "The service requires a .txt file")
        
    file = files[0]
    #file need to be outside directories.
    if "/" in file.filename:
        abort(400, "Subdirectories are not allowed")
    else:
        file.save(open(os.path.join(dataPath, file.filename), "wb"))
    
    filename = file.filename.split('.')[0]
    #generating the audio file
    wavfile.write(dataPath + "/" + filename + "_out.wav", data_config['sampling_rate'], audioFromEmbeds(filename, [embed]))
    
    #converting the audio file from .wav to .mp3
    in_fpath = dataPath + "/" + filename + "_out"
    subprocess.call(['ffmpeg', '-i', in_fpath + '.wav',
                   in_fpath + '.mp3'])
    
    #before returning the file, start a thread to delete the files.
    clearer = threading.Thread(target=cleanExample, args=(10, filename))
    clearer.start()
    
    #return the audio generated as an attachment
    return send_from_directory(dataPath, filename + "_out.mp3", as_attachment=True) 

#The create page handling post requests
@app.route('/audio/create', methods=['POST'])
def post_file():
    files = flask.request.files.getlist("file")
    receivedFiles = ""
    #debuging: print the files received
    for file in files:
        receivedFiles = receivedFiles + file.filename + "  "
    print("Received :", receivedFiles)
    
    #check for receiving only 2 files
    if len(files) != 2:
        print("More or less files received")
        abort(400, "The service requires a .mp3 and a .txt file")
    
    sharedFileName = files[0].filename.split('.')[0]
    #the files need to have the same name in order to be paired
    if sharedFileName != files[1].filename.split('.')[0]:
        print("Files do not share the same name")
        abort(400, "All files must have the same name" + sharedFileName)
    
    #files need to be outside directories.
    for file in files:
        if "/" in file.filename:
            abort(400, "Subdirectories are not allowed")
        else:
            file.save(open(os.path.join(dataPath, file.filename), "wb"))
    
    #generating the audio file         
    wavfile.write(dataPath + "/" + sharedFileName + "_out.wav", data_config['sampling_rate'], run_voiceCloning(sharedFileName))
    
    #converting the audio file from .wav to .mp3
    in_fpath = dataPath + "/" + sharedFileName + "_out"
    subprocess.call(['ffmpeg', '-i', in_fpath + '.wav',
                   in_fpath + '.mp3'])
    
    #before returning the file, start a thread to delete the files.
    clearer = threading.Thread(target=clean, args=(10, sharedFileName))
    clearer.start()
    
    #return the audio generated as an attachment
    return send_from_directory(dataPath, sharedFileName + "_out.mp3", as_attachment=True)

#Page that returns the ip or local ip of the device.
@app.route("/ip", methods=["GET"])
def getIp():
    return jsonify({'ip': request.remote_addr}), 200
    	
#Homepage
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

#MAIN    	
if __name__ == "__main__":
    setup()
    examplesSetup()
    app.run(host='192.168.0.112')
