FROM python:3.9.3
COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python download_models.py
CMD ["python", "start_smart_service.py"]