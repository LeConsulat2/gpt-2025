import streamlit as st
import subprocess
import tempfile
from pytube import YouTube
from pydub import AudioSegment
from openai import OpenAI


# Youtuve link에서 오디오 가져오기
def download_audio_from_youtube(youtube_url):
    """
    Downlaods audio from youtube video and saves it as a temp file.
    """
    youtube = YouTube(youtube_url)
    audio_stream = youtube.streams.filter(only_audio=True).first()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
        # Download audio to a temporary file
        audio_stream.download(output_path=None, filename=temp_audio_file.name)
        return temp_audio_file.name


# 비디오 파일에서 오디오 추출
def audio_extract(video_file):
    """
    Extracts audio from a video file using ffmpeg and saves it as a temporary mp3 file.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=False
    ) as extracted_audio_file:

        # Use ffmpeg to extract audio from the video
        ffmpeg_command = [
            "ffmpeg",
            "-i",
            video_file.name,
            "-vn",
            "-acodec",
            "libmp3lame",
            extracted_audio_file.name,
        ]
        subprocess.run(ffmpeg_command, check=True)


# Split audio into smaller 10-minute segments using for i in range
def split_audio_into_segments(audio_file_path, segment_duration_ms=10 * 60 * 1000):
    """
    Splits an audio file into smaller segments of a specified duration using for i in range.
    """
    full_audio_segment = AudioSegment.from_file(audio_file_path)
    audio_segments = []

    # Iterate over the audio in chunks of segment_duration_ms
    for i in range(0, len(full_audio_segment), segment_duration_ms):
        audio_segments.append(full_audio_segment[i : i + segment_duration_ms])

    return audio_segments


# Transcribe audio segments
def transcribe_audio(audio_segment_list):
    """
    Transcribes a list of audio segments using an OpenAI transcription client.
    """
    transcription_client = OpenAI()  # Initialize OpenAI client
    transcription_results = []

    for segment in audio_segment_list:
        with tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False
        ) as temp_audio_file:
            # Export audio segment to temporary file
            segment.export(temp_audio_file.name, format="mp3")

            # Send to transcription API
            with open(temp_audio_file.name, "rb") as audio_file:
                transcription_response = (
                    transcription_client.audio.transcriptions.create(
                        model="whisper-1", file=audio_file
                    )
                )
                transcription_results.append(transcription_response.text)

    # Combine all transcriptions into a single text
    return " ".join(transcription_results)


# Streamlit UI
st.title("Video and Audio Transcription Tool")

# User chooses input type: file upload or YouTube link
input_type = st.radio("Choose Input Type", options=["Upload File", "YouTube Link"])

# Handle file uploads
if input_type == "Upload File":
    uploaded_file = st.file_uploader(
        "Upload your video or audio file", type=["mp4", "mp3"]
    )

    if uploaded_file:
        with st.spinner("Processing uploaded file..."):
            file_type = uploaded_file.type

            if file_type == "audio/mpeg":  # MP3 file
                audio_segments = split_audio_into_segments(uploaded_file.name)
                transcription_result = transcribe_audio(audio_segments)
                st.success("Transcription completed!")
                st.text_area("Transcription:", transcription_result, height=300)

            elif file_type == "video/mp4":  # MP4 file
                extracted_audio_file = audio_extract(uploaded_file)
                audio_segments = split_audio_into_segments(extracted_audio_file)
                transcription_result = transcribe_audio(audio_segments)
                st.success("Transcription completed!")
                st.text_area("Transcription:", transcription_result, height=300)

            else:
                st.error("Unsupported file type. Please upload an MP4 or MP3 file.")

# Youtube Link
elif input_type == "YouTube Link":
    youtube_url = st.text_input("Enter Youtube video link:")

    if youtube_url:
        with st.spinner("Processing Youtube video... please wait"):
            downloaded_audio_file = download_audio_from_youtube(youtube_url)
            audio_segments = split_audio_into_segments(downloaded_audio_file)
            # Transcribe the audio segments
            transcription_result = transcribe_audio(audio_segments)
            st.success("Transcription completed!")
            st.text_area("Transcription:", transcription_result, height=300)
