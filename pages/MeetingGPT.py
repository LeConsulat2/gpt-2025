import streamlit as st
import os
import subprocess
import tempfile
import streamlit as st

from background import Black
from openai import OpenAI

Black.dark_theme()


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

# input_type = st.radio("Choose Input Type", options=["Upload File", "YouTube Link"])

uploaded_file = st.file_uploader("Upload your video or audio file", type=["mp4", "mp3"])

if uploaded_file:
    with st.spinner("Processing uploaded file..."):
        if uploaded_file.type == "audio/mpeg":  # MP3
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
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

# elif input_type == "YouTube Link":
#     youtube_url = st.text_input("Enter Youtube video link:")

# if youtube_url:
#     with st.spinner("Processing Youtube video..."):
#         audio_path = download_audio_from_youtube(youtube_url)
#         segments = split_audio_into_segments(audio_path)
#         transcription = transcribe_audio(segments)
#         st.success("Transcription completed!")
#         st.text_area("Transcription:", transcription, height=300)


# def download_audio_from_youtube(youtube_url):
#     """
#     유튜브 링크에서 오디오를 다운로드한 후 MP3로 변환합니다.
#     개선 사항:
#       - Windows 환경에서 NamedTemporaryFile을 사용해 파일 잠금 문제를 회피합니다.
#       - 다운로드된 파일의 크기를 확인하여 비정상적인 경우 에러를 발생시킵니다.
#       - ffmpeg 명령에 "-y" (강제 덮어쓰기)와 "-vn" (비디오 제거) 옵션을 추가합니다.
#     """
#     try:
#         # NamedTemporaryFile을 사용해 임시 파일 생성 (delete=False로 파일 잠금 문제 회피)
#         temp_file = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
#         temp_path = temp_file.name
#         temp_file.close()  # 파일을 닫아서 외부 프로세스(ffmpeg 등)에서 접근할 수 있도록 함

#         ydl_opts = {
#             "format": "bestaudio/best",
#             "outtmpl": temp_path,
#             "quiet": False,
#         }

#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             st.write("Attempting to extract video info...")
#             info = ydl.extract_info(youtube_url, download=True)
#             st.write("Video title:", info.get("title", "Unknown"))
#             st.write("Download completed")

#         # 추가 디버깅: 다운로드된 파일 크기를 출력
#         file_size = os.path.getsize(temp_path)
#         st.write("Downloaded file size:", file_size, "bytes")
#         if file_size < 1024:
#             st.error("Downloaded file appears to be too small or corrupted")
#             raise Exception("Downloaded file is too small")

#         # 결과로 변환할 MP3 파일의 경로 결정
#         output_path = temp_path.replace(".webm", ".mp3")

#         st.write("Converting to MP3...")
#         # ffmpeg 명령어 구성: -y (덮어쓰기), -vn (비디오 스트림 제거), libmp3lame로 인코딩
#         command = [
#             "ffmpeg",
#             "-y",
#             "-v",
#             "verbose",
#             "-i",
#             temp_path,
#             "-vn",
#             "-acodec",
#             "libmp3lame",
#             output_path,
#         ]

#         result = subprocess.run(command, capture_output=True, text=True)
#         st.write("FFmpeg command:", " ".join(command))
#         st.write("FFmpeg stdout:")
#         st.code(result.stdout)
#         st.write("FFmpeg stderr:")
#         st.code(result.stderr)

#         if result.returncode != 0:
#             raise subprocess.CalledProcessError(
#                 result.returncode, command, result.stdout, result.stderr
#             )

#         st.write("Conversion completed")
#         return output_path

#     except Exception as e:
#         st.error("Error: " + str(e))
#         try:
#             if os.path.exists(temp_path):
#                 os.unlink(temp_path)
#         except Exception as ex:
#             st.error("Failed to delete temporary file: " + str(ex))
#         raise
