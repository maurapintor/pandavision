FROM  python:3.7

ENV API_SERVER_HOME=/opt/www
WORKDIR "$API_SERVER_HOME"
COPY "./requirements.txt" "./"
    
RUN python -m pip install -r requirements.txt

COPY "./" "./"

ENV SHARED_DATA_FOLDER=/opt/data

ENV PYTHONPATH "${PYTHONPATH}:/opt/data"
    
#USER nobody
CMD [ "python", "__init.py__" ]
