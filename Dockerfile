FROM  python:3.7

COPY "./requirements.txt" "./"

RUN python -m pip install -r requirements.txt

WORKDIR /tmp/
COPY "./" "./"

RUN python -m pip install .