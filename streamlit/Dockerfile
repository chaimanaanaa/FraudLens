FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir streamlit==1.35.0 Pillow==10.3.0 requests==2.32.2 python-dotenv==1.0.1

COPY streamlit/ ./streamlit/

EXPOSE 8501

CMD ["streamlit", "run", "streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--theme.base=light", "--theme.primaryColor=#1D4ED8", "--server.maxUploadSize=15"]
