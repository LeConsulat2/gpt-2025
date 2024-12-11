import streamlit as st
import subprocess
import math
import glob
import os
from openai import OpenAI
from pydub import AudioSegment
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import yt_dlp
import tempfile


st.set_page_config(
    page_title="MeetingGPT",
    page_icon="📃",
)


# OpenAI 클라이언트 초기화
client = OpenAI()

st.title("MeetingGPT")

st.markdown(
    """
    ## Welcome to MeetingGPT

    Upload a video file or provide a YouTube link for:
    - Detailed transcription
    - Concise summary
    - Q&A about the content
    
    Get started by choosing your input method in the sidebar.
    """
)

# LangChain 모델 초기화
llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
)


def download_youtube_audio(youtube_url):
    """YouTube 영상에서 오디오 추출"""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": temp_audio.name,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "prefer_ffmpeg": True,
            "quiet": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            return temp_audio.name
        except Exception as e:
            st.error(f"YouTube 다운로드 오류: {str(e)}")
            return None


@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    """오디오 청크 변환"""
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
            text_file.write(transcript.text)


@st.cache_data()
def extract_audio_from_video(video_path):
    """비디오에서 오디오 추출"""
    audio_path = video_path.replace("mp4", "mp3")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        audio_path,
    ]
    subprocess.run(command)
    return audio_path


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    """오디오를 청크로 분할"""
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(
            f"{chunks_folder}/chunk_{i}.mp3",
            format="mp3",
        )


# 사이드바에서 입력 방식 선택
with st.sidebar:
    input_type = st.radio("Choose Input Type", ["Upload Video", "YouTube Link"])

    if input_type == "Upload Video":
        video = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mkv", "mov"],
        )
    else:
        youtube_url = st.text_input("Enter YouTube URL:")

# 메인 처리 로직
if input_type == "Upload Video" and video:
    process_input = video
elif input_type == "YouTube Link" and youtube_url:
    process_input = youtube_url
else:
    process_input = None

if process_input:
    chunks_folder = f"./.cache/chunks_{os.path.splitext(str(process_input))[-1]}"
    transcription_path = (
        f"./.cache/transcription_{os.path.splitext(str(process_input))[-1]}.txt"
    )

    if not os.path.exists(transcription_path):
        with st.spinner("Processing..."):
            os.makedirs("./.cache", exist_ok=True)

            if input_type == "Upload Video":
                video_content = process_input.read()
                video_path = f"./.cache/{process_input.name}"
                with open(video_path, "wb") as f:
                    f.write(video_content)
                st.info("Extracting audio...")
                audio_path = extract_audio_from_video(video_path)
            else:  # YouTube Link
                st.info("Downloading YouTube audio...")
                audio_path = download_youtube_audio(process_input)
                if not audio_path:
                    st.error("Failed to download YouTube video")
                    st.stop()

            st.info("Cutting audio segments...")
            os.makedirs(chunks_folder, exist_ok=True)
            cut_audio_in_chunks(audio_path, 10, chunks_folder)

            st.info("Transcribing audio...")
            transcribe_chunks(chunks_folder, transcription_path)

    # 탭 생성
    transcription_tab, summary_tab, qa_tab = st.tabs(
        ["Transcription", "Summary", "Q&A"]
    )

    # 변환 텍스트 표시
    with transcription_tab:
        if os.path.exists(transcription_path):
            with open(transcription_path, "r", encoding="utf-8") as file:
                st.write(file.read())

    # 요약 생성
    with summary_tab:
        if st.button("Generate summary"):
            loader = TextLoader(transcription_path)
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=800,
                chunk_overlap=100,
            )
            docs = loader.load_and_split(text_splitter=splitter)

            if docs:
                first_summary_prompt = ChatPromptTemplate.from_template(
                    """
                    Write a concise summary of the following:
                    "{text}"
                    CONCISE SUMMARY:                
                    """
                )

                first_summary_chain = first_summary_prompt | llm | StrOutputParser()

                summary = first_summary_chain.invoke({"text": docs[0].page_content})

                refine_prompt = ChatPromptTemplate.from_template(
                    """
                    Your job is to produce a final summary.
                    We have provided an existing summary up to a certain point: {existing_summary}
                    We have the opportunity to refine the existing summary (only if needed) with some more context below.
                    ------------
                    {context}
                    ------------
                    Given the new context, refine the original summary.
                    If the context isn't useful, RETURN the original summary.
                    """
                )

                refine_chain = refine_prompt | llm | StrOutputParser()

                with st.spinner("Summarizing..."):
                    for i, doc in enumerate(docs[1:]):
                        st.info(f"Processing document {i+1}/{len(docs)-1}")
                        summary = refine_chain.invoke(
                            {
                                "existing_summary": summary,
                                "context": doc.page_content,
                            }
                        )
                    st.write(summary)

    # Q&A 섹션
    with qa_tab:
        st.write("Coming soon...")
