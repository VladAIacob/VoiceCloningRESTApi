import flask
from flask import request, jsonify, abort, send_from_directory
from scipy.io import wavfile
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import subprocess
import threading
import librosa
import time
import os

app = flask.Flask(__name__)
app.config["DEBUG"] = True

#Data paths
examplePath = "audio/examples"
dataPath = "audio/data"
url = "test.wav"
synthesizer = ""

#Example embeds
barackobama = ""
gordonRamsay = ""
stephenHawking = ""

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
    global synthesizer
    encoder_weights = Path("encoder/saved_models/pretrained.pt")
    vocoder_weights = Path("vocoder/saved_models/pretrained/pretrained.pt")
    syn_dir = Path("synthesizer/saved_models/logs-pretrained/taco_pretrained")
    encoder.load_model(encoder_weights)
    synthesizer = Synthesizer(syn_dir)
    vocoder.load_model(vocoder_weights)

#Generate audio from embeds
def audioFromEmbeds(filename, embed):
    textPath = dataPath + "/" + filename + ".txt"
    #reading the text file and prepare the text string for the synthesizer
    textFile = open(textPath)
    text = textFile.read().replace("\n", " ")
    textFile.close()
    
    #synthesize the text together with the embeds
    specs = synthesizer.synthesize_spectrograms([text], [embed])
  
    #generate the audio using the vocoder
    generated_wav = vocoder.infer_waveform(specs[0])
    
    return np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

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
    
    return audioFromEmbeds(filename, embed)

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
    wavfile.write(dataPath + "/" + filename + "_out.wav", synthesizer.sample_rate, audioFromEmbeds(filename, embed))
    
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
    wavfile.write(dataPath + "/" + sharedFileName + "_out.wav", synthesizer.sample_rate, run_voiceCloning(sharedFileName))
    
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
