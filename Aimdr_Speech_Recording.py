import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 10  # Recording duration

print("Recording...")
audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()
write("meeting_audio.wav", fs, audio)
print("Recording saved")

#recording for 5 times

'''meeting_running = True
chunk_count = 0

print("Meeting transcription started... Ctrl+C to stop")

try:
    while meeting_running:
        filename = f"chunk_{chunk_count}.wav"

        record_audio_chunk(filename, duration=5)
        text = whisper_transcribe(filename)

        if text.strip():
            save_to_database(text)

        os.remove(filename)  # cleanup
        chunk_count += 1

except KeyboardInterrupt:
    print("\nMeeting stopped")'''


