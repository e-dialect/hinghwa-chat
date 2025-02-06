FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -ihttps://mirrors.aliyun.com/pypi/simple/ -U openai 
RUN pip install -ihttps://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

COPY src/ /app/src
COPY data/ /app/data

ENV DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY}

RUN python src/load_px_words.py

EXPOSE 8000

CMD ["python", "src/ui_server.py"]
