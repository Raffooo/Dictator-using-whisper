import numpy as np
import whisper
import sounddevice as sd
import wavio
import threading
from pynput import keyboard
from pynput.keyboard import Controller
import time
import os
import tkinter as tk
import pystray
from pystray import MenuItem as item
from PIL import Image
import sys
import warnings


if not os.path.exists("recordings"):
    os.makedirs("recordings")

recording = False
recordingThread = None
stopRecording = threading.Event()
keyboardPresser = Controller()
# model = whisper.load_model("turbo", device="cuda")
# ^ comment to save RAM while not transcribing ^

# use tkinter to add a red 'recording' icon in the top left
root = tk.Tk()
root.overrideredirect(True)
root.attributes("-topmost", True)  # always on top
root.configure(bg="white")  # white background that is then made transparent
root.wm_attributes("-transparentcolor", "white")

# position in the top left corner
root.geometry("+0+0")

red_dot = tk.Label(root, text="●", fg="red", bg="white", font=("Arial", 30))
red_dot.pack()

root.withdraw()

def showRecordingIndicator():
    root.after(0, lambda: root.deiconify())

def hideRecordingIndicator():
    root.after(0, lambda: root.withdraw())


#####################################################################


def simulateKeypress(key):
    keyboardPresser.press(key)
    keyboardPresser.release(key)
    time.sleep(0.01) # to give a nice 'typing' effect


def transcribeAudio():
    # global model
    # # print("Loading model")
    # startTime = time.time()
    warnings.filterwarnings("ignore", category=FutureWarning)
    model = whisper.load_model("turbo", device="cuda")
    # ^ uncomment to save RAM when not transcribing ^
    # print(f"Model loaded in {time.time() - startTime} seconds")

    # startTime = time.time()
    # print("\nStarting transcription")
    transcription = model.transcribe(f"recordings/output.wav")
    # print(f"Transcribed in {time.time() - startTime} seconds")
    # print(transcription["text"] + "\n\n")

    # simulate each key press in the transcription
    for char in transcription["text"]:
        simulateKeypress(char)


def recordAudio():
    # print("Started recording")
    global recording
    sampleRate = 44100
    channels = 1

    # show the red dot while recording
    showRecordingIndicator()

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
        # print("recording complete")

    recording = False
    stopRecording.clear()
    hideRecordingIndicator()

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
        # # print("Listening for ctrl+\\ to toggle recording")
        listener.join()


threading.Thread(target=main, daemon=True).start()

# make an icon in the system tray

def resourcePath(relativePath):
    """ Get the absolute path to a resource, works for dev and for PyInstaller. """
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller extracts to _MEIPASS
        return os.path.join(sys._MEIPASS, relativePath)
    return os.path.join(os.path.abspath("."), relativePath)

# Use user's own image (ensure 'icon.png' is placed in the same directory)
trayImage = Image.open(resourcePath("icon.png"))

def onQuit(icon, item):
    icon.stop()
    root.quit()

icon = pystray.Icon(
    "Dictator",
    trayImage,
    "Speak, and I will type.",
    menu=pystray.Menu(
        item("Quit", onQuit)
    )
)

icon.run_detached()

root.mainloop()
