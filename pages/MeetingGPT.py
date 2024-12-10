import streamlit as st
import subprocess


def audio_extract(video):
    output_filename = video.name.split(".", 1)[0] + ".mp3"
    command = [
        "ffmpeg",
        "-i",
        video,
        "-vn",
        "-acodec",
        "libmp3lame",
        "output.mp3",
    ]
    subprocess.run(command, check=True)
    return output_filename


# command = [
#     "ffmpeg",
#     "-i", video,          # Input file
#     "-vn",               # 'Video No' - Skip the video stream
#     "-acodec",          # Specify which Audio CODEC to use
#     "libmp3lame",       # LAME is a high-quality MP3 encoder
#     "output.mp3"        # Output file
# ]

uploaded_files = st.file_uploader("Upload your video", type=["mp4"])

if uploaded_files:
    with st.spinner("Processing video... please wait"):
        output_file = audio_extract(uploaded_files)
        st.success(f"Audio Extracted: {output_file} successfully")


def audiosegment()

def audio_cut(output_file):
    audio = audiosegment.from_mp3(output_file)
    audio.duration_minutes


