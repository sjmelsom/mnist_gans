FROM pytorch:19.09-py3

COPY toy_gans /opt/

# RUN cd /opt/toy_gans

# RUN pip install -r requirements.txt

ENTRYPOINT ["bash"]
