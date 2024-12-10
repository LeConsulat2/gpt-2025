import streamlit as st
import subprocess
import tempfile
from pydub import AudioSegment
from openai import OpenAI


def audio_extract(video_file):
    # Create temporary files for both the uploaded video and extracted audio
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video_file:
        # Write the uploaded video to a temporary file
        temp_video_file.write(video_file.read())
        temp_video_file.flush()

        with tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False
        ) as extracted_audio_file:
            # Use ffmpeg to extract audio from the video
            ffmpeg_command = [
                "ffmpeg",
                "-i",
                temp_video_file.name,
                "-vn",
                "-acodec",
                "libmp3lame",
                extracted_audio_file.name,
            ]
            subprocess.run(ffmpeg_command, check=True)

            # Load the full audio file as an AudioSegment object
            full_audio_segment = AudioSegment.from_mp3(extracted_audio_file.name)

            # Define the length for each audio segment (10 minutes in milliseconds)
            segment_duration_ms = 10 * 60 * 1000

            # Divide the full audio into smaller segments
            # List comprehension slices the audio into chunks of 10 minutes
            # Example for a 25-minute audio:
            # - First iteration: 0 ms to 600,000 ms (0–10 mins)
            # - Second iteration: 600,000 ms to 1,200,000 ms (10–20 mins)
            # - Third iteration: 1,200,000 ms to the end (20–25 mins)
            audio_segments = [
                full_audio_segment[start_index : start_index + segment_duration_ms]
                for start_index in range(
                    0, len(full_audio_segment), segment_duration_ms
                )
            ]

            # Returns a list of audio segments
            # Each segment is 10 minutes long, except the last one, which may be shorter
            return audio_segments


def transcribe_audio(audio_segment_list):
    # Initialize the OpenAI transcription client
    transcription_client = OpenAI()
    transcription_results = []

    # Process each audio segment individually
    for individual_audio_segment in audio_segment_list:
        # Create a temporary file for the audio segment
        with tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False
        ) as temp_audio_file:
            # Export the audio segment to the temporary file
            individual_audio_segment.export(temp_audio_file.name, format="mp3")

            # Open the temporary file and send it to the transcription service
            with open(temp_audio_file.name, "rb") as audio_file_for_transcription:
                transcription_response = (
                    transcription_client.audio.transcriptions.create(
                        model="whisper-1", file=audio_file_for_transcription
                    )
                )

            # Append the transcribed text to the results list
            transcription_results.append(transcription_response.text)

    # Combine all transcriptions into a single text string
    return " ".join(transcription_results)


# File uploader for users to upload their video file
uploaded_files = st.file_uploader("Upload your video", type=["mp4"])

if uploaded_files:
    # Show a spinner while processing the video
    with st.spinner("Processing video... please wait"):
        # Extract the audio segments from the uploaded video
        extracted_audio_segments = audio_extract(uploaded_files)

        # Display a success message after extraction is complete
        st.success(
            f"Audio extracted successfully. Total segments: {len(extracted_audio_segments)}"
        )
