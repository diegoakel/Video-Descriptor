import json
from flask import Flask, render_template, request
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
import shutil
import io
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import cv2
import base64
import glob

# load_dotenv()

app = Flask(__name__)

# client = OpenAI(api_key=os.getenv("API_KEY"))


def make_chunks(audio, chunk_length_ms):
    """
    Breaks an audio file into chunks of a specified length
    """
    chunks = []
    while len(audio) > chunk_length_ms:
        chunks.append(audio[:chunk_length_ms])
        audio = audio[chunk_length_ms:]
    chunks.append(audio)
    return chunks

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/getfile", methods=["POST"])
def getfile():
    video_file = request.files["videoFile"]
    api_key = request.form["apiKey"]
    use_transcription = request.form.get("useTranscription") == "true"
    want_voiceover = request.form.get("wantVoiceover") == "true"
    frame_usage_rate = int(request.form.get("framesInput"))
    narration_style = request.form.get("narrationStyle")

    client = OpenAI(api_key=api_key)


    video_file.save("temp_video.mp4")

    # Extract frames from the video
    cap = cv2.VideoCapture("temp_video.mp4")
    i = 0
    base64Frames = []

    # frame_usage_rate = 500

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_usage_rate == 0:  # Extract every 50th frame
            cv2.imwrite(f"frame{i}.jpg", frame)
            with open(f"frame{i}.jpg", "rb") as img_file:
                base64Frames.append(base64.b64encode(img_file.read()).decode('utf-8'))
        i += 1
    cap.release()

    with VideoFileClip("temp_video.mp4") as clip:
        audio = clip.audio
        audio.write_audiofile("audio.wav")
        clip.close()  # Close the VideoFileClip object


    # Transcribe the audio
    if use_transcription:
        audio = AudioSegment.from_wav("audio.wav")
        chunk_length_ms = 60000  # length of each audio chunk (in milliseconds)
        chunks = make_chunks(audio, chunk_length_ms)
        transcripts = []

        for i, chunk in enumerate(chunks):
            chunk_name = f"chunk{i}.wav"
            chunk.export(chunk_name, format="wav")
            with open(chunk_name, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=f,
                    response_format="text"
                )
            transcripts.append(transcript)

    prompt = f"""
    "I'm giving you frames of a video. Describe a very good and complete description for it in the style of {narration_style}. Dont describe it frame by frame, but use all of the kwonledge and then describe in a continuous big paragraph or 2.
    """

    if use_transcription:
        prompt += f"You can also use the audio transcript to help you: {transcripts}"

    
    # Generate descriptions for the frames
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                prompt,
                *map(lambda x: {"image": x, "resize": 768}, base64Frames),
            ],
        },
    ]
    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 700,
    }
    result = client.chat.completions.create(**params)
    frame_descriptions = result.choices[0].message.content

    # delete temporary files after use
    os.remove("temp_video.mp4")

    if use_transcription:
        os.remove("audio.wav")
        for i in range(len(chunks)):
            os.remove(f"chunk{i}.wav")

    for i in range(0, len(base64Frames)*frame_usage_rate, frame_usage_rate):
        os.remove(f"frame{i}.jpg")

    files = glob.glob('static/*')
    for f in files:
        os.remove(f)


    voiceover_file = None
    if want_voiceover:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=frame_descriptions
        )
        voiceover_file = "voiceover.mp3"
        response.stream_to_file("static/" + voiceover_file)



    return render_template("index.html", frame_descriptions=frame_descriptions, voiceover_file=voiceover_file)

if __name__ == "__main__":
    app.run(debug=True)
