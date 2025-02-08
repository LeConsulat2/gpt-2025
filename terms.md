'''
임베딩 생성 및 문서 분할

AUT 비유:
AUT 대학의 도서관에서는 다양한 학습 자료(문서)를 학생들이 쉽게 검색할 수 있도록
각 교재의 내용을 디지털 자료(벡터)로 변환(임베딩 생성)한 후, 각 챕터나 섹션(청크)으로
나누어 관리합니다.

```Python
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
```

긴 문서를 일정한 크기의 청크로 나누는 작업
chunk_size=1000: 각 청크의 최대 문자 수
chunk_overlap=100: 인접 청크들 간 100자씩 중복되어 문맥 연결 유지

```Python
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(documents)
```

벡터 저장소(Vector Store)와 검색자(Retriever) 생성

AUT 비유:
AUT 대학교 도서관 시스템에서는 모든 학습 자료를 색인화(indexing)하여,
학생이 필요한 정보를 빠르게 찾아낼 수 있도록 합니다.

```Python
vectorstore = FAISS.from_documents(splits, embeddings)
```

사용자의 질문과 관련된 문서를 신속하게 검색하는 검색자 생성

```Python
retriever = vectorstore.as_retriever()
```
