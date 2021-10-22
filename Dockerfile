FROM  python:3.7

ENV API_SERVER_HOME=/opt/www
WORKDIR "$API_SERVER_HOME"
COPY "./requirements.txt" "./"

RUN set -ex; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends libgtk2.0-dev libsndfile1 ffmpeg
    
RUN pip3 install -r requirements.txt

COPY "./" "./"

ENV MONGO_DB_NAME=aloha-test \
    MONGO_DB_TCP_ADDR=mongo \
    MONGO_DB_PORT=27017 \
    SHARED_DATA_FOLDER=/opt/data \
    PYTHONPATH="${PYTHONPATH}:/opt/www:/opt/www/onnxparser"

ENV PYTHONPATH "${PYTHONPATH}:/opt/data"
    
#USER nobody
CMD [ "python3", "__init.py__" ]
