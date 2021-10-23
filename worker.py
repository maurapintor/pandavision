"""
For more workers use the following command:

docker-compose up --scale worker=2
"""

import os

import redis
from rq import Worker, Queue, Connection

# use rq-dashboard for visualization
listen = ['sec-evals', 'adv-gen']

REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = os.environ.get('REDIS_PORT', '6378')
conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

if __name__ == '__main__':
    while True:
        try:
            with Connection(conn):
                q = Queue()
                worker = Worker(list(map(Queue, listen)))
                worker.work()
        except:
            pass
        else:
            break
