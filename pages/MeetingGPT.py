import os
import time
import subprocess
import tempfile
import threading
import streamlit as st
from background import Black
from openai import OpenAI

Black.dark_theme()

# 세션 상태에 stop 변수 추가 (초기값 False)
if "stop" not in st.session_state:
    st.session_state["stop"] = False


def run_command_with_stop(command):
    """
    ffmpeg 같은 외부 명령어를 실행하되, 중간에 st.session_state["stop"]가
    True가 되면 프로세스를 종료합니다.
    """
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    while process.poll() is None:
        if st.session_state["stop"]:
            process.terminate()
            return None, "Terminated by user"
        time.sleep(0.1)
    stdout, stderr = process.communicate()
    return stdout, stderr


def audio_extract(video_file):
    """
    비디오에서 오디오(WAV) 추출
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        temp_video.write(video_file.read())
        temp_video.flush()

    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        command = [
            "ffmpeg",
            "-i",
            temp_video.name,
            "-vn",
            "-acodec",
            "pcm_s16le",  # WAV 변환
            "-ar",
            "16000",  # 샘플링 레이트 16kHz
            "-ac",
            "1",  # 모노 채널
            temp_audio.name,
        ]
        stdout, stderr = run_command_with_stop(command)
        if stdout is None:
            raise Exception("Audio extraction stopped by user")
        return temp_audio.name
    finally:
        os.remove(temp_video.name)  # 사용 후 삭제


def split_audio_into_segments(audio_file_path, segment_duration=600):
    """
    오디오 파일을 10분 단위로 분할 (ffmpeg 사용)
    """
    segments = []
    temp_dir = tempfile.mkdtemp()

    try:
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
            f"{temp_dir}/segment_%03d.wav",
        ]
        stdout, stderr = run_command_with_stop(command)
        if stdout is None:
            return []  # 중단된 경우 빈 리스트 반환

        for segment_file in sorted(os.listdir(temp_dir)):
            if segment_file.endswith(".wav"):
                if st.session_state["stop"]:  # 중지 버튼 체크
                    return []
                full_path = os.path.join(temp_dir, segment_file)
                segments.append(full_path)

        return segments
    finally:
        os.remove(audio_file_path)


def transcribe_audio(audio_segment_paths):
    """
    오디오 세그먼트들을 텍스트로 변환
    """
    client = OpenAI()
    transcription_results = []

    for segment_path in audio_segment_paths:
        if st.session_state["stop"]:  # 중지 버튼 체크
            return ""
        with open(segment_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
            transcription_results.append(transcription.text)
        os.remove(segment_path)

    return " ".join(transcription_results)


def process_file(uploaded_file):
    """파일 처리 및 변환 실행"""
    if uploaded_file.type == "audio/wav":  # WAV 파일인 경우
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(uploaded_file.read())
            temp_audio.flush()
            segments = split_audio_into_segments(temp_audio.name)
            if st.session_state["stop"]:
                return
            transcription = transcribe_audio(segments)
            st.success("Transcription completed!")
            st.text_area("Transcription:", transcription, height=300)

    elif uploaded_file.type == "video/mp4":  # MP4 → WAV 변환 후 처리
        audio_path = audio_extract(uploaded_file)
        segments = split_audio_into_segments(audio_path)
        if st.session_state["stop"]:
            return
        transcription = transcribe_audio(segments)
        st.success("Transcription completed!")
        st.text_area("Transcription:", transcription, height=300)


# Streamlit UI
st.title("Video and Audio Transcription Tool")
uploaded_file = st.file_uploader("Upload your video or audio file", type=["mp4", "wav"])

if uploaded_file:
    if st.button("Start Processing"):
        st.session_state["stop"] = False  # 중지 상태 초기화
        thread = threading.Thread(target=process_file, args=(uploaded_file,))
        thread.start()

if st.button("Stop Processing"):
    st.session_state["stop"] = True  # 중지 상태 활성화
    st.warning("Processing Stopped!")
