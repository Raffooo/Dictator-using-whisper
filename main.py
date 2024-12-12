import numpy as np
import whisper
import sounddevice as sd
import wavio
import threading
from pynput import keyboard
from pynput.keyboard import Controller
import time

recording = False
recordingThread = None
stopRecording = threading.Event()
keyboardPresser = Controller()
model = whisper.load_model("turbo", device="cuda")


def simulateKeypress(key):
    keyboardPresser.press(key)
    keyboardPresser.release(key)


def transcribeAudio():
    global model
    print("Loading model")
    startTime = time.time()
    # model = whisper.load_model("turbo", device="cuda")
    print(f"Model loaded in {time.time() - startTime} seconds")

    startTime = time.time()
    print("\nStarting transcription")
    transcription = model.transcribe(f"recordings/output.wav")
    print(f"Transcribed in {time.time() - startTime} seconds")
    print(transcription["text"] + "\n\n")

    # simulate each key press in the transcription
    for char in transcription["text"]:
        simulateKeypress(char)


def recordAudio():
    print("Started recording")
    global recording
    sampleRate = 44100
    channels = 1

    buffer = []  # buffer to save the recording chunks to

    # function that is called every 100ms (every time the stream refreshes)
    # raises an exception if stopRecording is set
    def callback(indata, frames, time, status):
        if stopRecording.is_set():
            raise sd.CallbackStop()
        buffer.append(indata.copy())

    # Record audio continuously until that exception is raised
    try:
        with sd.InputStream(samplerate=sampleRate, channels=channels, callback=callback, dtype="int16"):
            while not stopRecording.is_set():
                sd.sleep(100)

    except sd.CallbackStop:
        pass

    # put all the chunks together and save
    if buffer:
        fullRecording = np.concatenate(buffer)
        wavio.write(f"recordings/output.wav", fullRecording, rate=44100, sampwidth=2)
        print("recording complete")

    recording = False
    stopRecording.clear()

    # start a transcription thread
    transcriptionThread = threading.Thread(target=transcribeAudio)
    transcriptionThread.start()


def main():
    global recording, recordingThread
    currentKeys = set()
    startRecordingCombo = {keyboard.Key.alt_l, keyboard.KeyCode(char="\\")}

    # when any key is pressed
    def onPress(key):
        global recording, recordingThread
        currentKeys.add(key)

        # if the combination of keys pressed is the shortcut
        if startRecordingCombo.issubset(currentKeys):
            # if not already recording
            if not recording:
                recording = True
                stopRecording.clear()
                recordingThread = threading.Thread(target=recordAudio)
                recordingThread.start()

            else:
                stopRecording.set()

    def onRelease(key):
        currentKeys.discard(key)

    with keyboard.Listener(on_press=onPress, on_release=onRelease) as listener:
        print("Listening for ctrl+\\ to toggle recording")
        listener.join()


main()