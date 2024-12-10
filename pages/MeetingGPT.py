import streamlit as st
import subprocess
import tempfile
from pytube import YouTube
import yt_dlp
from openai import OpenAI
import os
from background import Black

Black.dark_theme()


# pytube 대신 yt-dlp 사용
import yt_dlp


def download_audio_from_youtube(youtube_url):
    """
    유튜브 링크에서 오디오 다운로드 (yt-dlp 사용)
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": temp_audio_file.name,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                }
            ],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        return temp_audio_file.name


def audio_extract(video_file):
    """
    비디오에서 오디오 추출
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        temp_video.write(video_file.read())
        temp_video.flush()

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            command = [
                "ffmpeg",
                "-i",
                temp_video.name,
                "-vn",
                "-acodec",
                "libmp3lame",
                temp_audio.name,
            ]
            subprocess.run(command, check=True)
            return temp_audio.name


def split_audio_into_segments(audio_file_path, segment_duration=600):  # 10분 = 600초
    """
    오디오 파일을 10분 단위로 분할 (ffmpeg 사용)
    """
    segments = []
    with tempfile.TemporaryDirectory() as temp_dir:
        # ffmpeg로 오디오 분할
        command = [
            "ffmpeg",
            "-i",
            audio_file_path,
            "-f",
            "segment",
            "-segment_time",
            str(segment_duration),
            "-c",
            "copy",
            f"{temp_dir}/segment_%03d.mp3",
        ]
        subprocess.run(command, check=True)

        # 분할된 파일들 수집
        for segment_file in sorted(os.listdir(temp_dir)):
            if segment_file.endswith(".mp3"):
                full_path = os.path.join(temp_dir, segment_file)
                segments.append(full_path)

    return segments


def transcribe_audio(audio_segment_paths):
    """
    오디오 세그먼트들을 텍스트로 변환
    """
    client = OpenAI()
    transcription_results = []

    for segment_path in audio_segment_paths:
        with open(segment_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
            transcription_results.append(transcription.text)

    return " ".join(transcription_results)


# Streamlit UI는 그대로 유지
st.title("Video and Audio Transcription Tool")

input_type = st.radio("Choose Input Type", options=["Upload File", "YouTube Link"])

if input_type == "Upload File":
    uploaded_file = st.file_uploader(
        "Upload your video or audio file", type=["mp4", "mp3"]
    )

    if uploaded_file:
        with st.spinner("Processing uploaded file..."):
            if uploaded_file.type == "audio/mpeg":  # MP3
                with tempfile.NamedTemporaryFile(
                    suffix=".mp3", delete=False
                ) as temp_audio:
                    temp_audio.write(uploaded_file.read())
                    temp_audio.flush()
                    segments = split_audio_into_segments(temp_audio.name)
                    transcription = transcribe_audio(segments)
                    st.success("Transcription completed!")
                    st.text_area("Transcription:", transcription, height=300)

            elif uploaded_file.type == "video/mp4":  # MP4
                audio_path = audio_extract(uploaded_file)
                segments = split_audio_into_segments(audio_path)
                transcription = transcribe_audio(segments)
                st.success("Transcription completed!")
                st.text_area("Transcription:", transcription, height=300)

elif input_type == "YouTube Link":
    youtube_url = st.text_input("Enter Youtube video link:")

    if youtube_url:
        with st.spinner("Processing Youtube video..."):
            audio_path = download_audio_from_youtube(youtube_url)
            segments = split_audio_into_segments(audio_path)
            transcription = transcribe_audio(segments)
            st.success("Transcription completed!")
            st.text_area("Transcription:", transcription, height=300)
