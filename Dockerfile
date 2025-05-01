FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY books/ ./books/
COPY cybersec_chatbot.py ./
COPY ggml-gpt4all-j-v1.3-groovy.bin ./
ENTRYPOINT ["python","cybersec_chatbot.py"]
