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
REDIS_PORT = os.environ.get('REDIS_PORT', '6379')
conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

def my_handler(job, *exc_info):
    print('\tFailed job handler at RPI for Job: %s' % str(job))
    import requests
    url='http://%s/api/dse_engine/failed?job_id=%s'%('dse_engine:5000',job.id)

    r = requests.post(url,data={"module": "sec_en"})
    print(r.json())
    return True

if __name__ == '__main__':
    while True:
        try:
            with Connection(conn):
                q = Queue()

                worker = Worker(list(map(Queue, listen)))
                worker.push_exc_handler(my_handler)
                worker.work()
        except:
            pass
        else:
            break
