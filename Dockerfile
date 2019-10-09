FROM pytorch:19.09

COPY toy_gans /

RUN pip install -r requirements.txt

