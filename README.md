# :rocket: :panda_face: :fire: PandaVision
Integrate and automate security evaluations with onnx, pytorch, and SecML!
    
## Installation

### Starting the server without Docker

If you want to run the server with docker, skip to the [next section](#starting-the-server-with-docker).

This project uses [Redis-RQ](http://python-rq.org/) for handling the queue of requested jobs. 
Please install [Redis](https://redis.io/) if you plan to run this Flask server without using Docker.

Then, install the Python requirements, running the following command in your shell:

```bash
pip install -r requirements.txt
```

Make sure your Redis server is running on your local machine. 
Test the Redis connection with the following command:

```bash
redis-cli ping
```

The response `PONG` should appear in the shell.

If the database servers is down, check the linked docs for finding out how to restart it in your system.

Notice: the code is expected to connect to the database through its default port, 6379 for Redis. 

Now we are ready to start the server. Don't forget that this system uses external workers 
to process the long-running tasks, so we need to start the workers along with the sever. 
Run the following commands from the `app` folder:

```bash
python3 worker.py
```

Now open another shell and run the server:

```bash
python3 app/runserver.py
```

### Starting the server with docker

If you already started the server locally, you can skip to the [next section](#usage).

If you already started the server locally, *but you want to start it with docker instead*, you should stop the 
running services. On linux, press `CTRL + C` to stop the server and the worker, then stop the redis service on the 
machine.

```bash
sudo service redis stop
```

In order to use the docker-compose file provided, install [Docker](https://www.docker.com/) and start the Docker service.

Since this project uses different interconnected containers, it is recommended to install and 
use [Docker Compose](https://docs.docker.com/compose/).

Once set up, Docker Compose will automatically take care of the setup process. 
Just type the following commands in your shell, from the `app` path:

```bash
docker-compose build
docker-compose up
```

If you want to use more workers, the following command should be used(replace 
the number `2` with the number of workers you want to set up):

```bash
docker-compose up --scale worker=2
```

## Usage

### Quick start

For a demo example, you can download a 
[sample containing few images of the imagenet dataset](https://github.com/maurapintor/pandavision/releases/download/v0.1/data.h5)
and a [resnet50-pretrained model from the onnx zoo](https://github.com/maurapintor/pandavision/releases/download/v0.1/model.onnx).

And you can place the data in a directory called `appdata`, in the main root of the repository.

### Supported models

You can export your own [ONNX](https://github.com/onnx/tutorials) pretrained model from the library of your choice, 
and pass them to the module. 
This project uses [onnx2pytorch](https://github.com/ToriML/onnx2pytorch) as a dependency to load the ONNX models.
Check out [the supported operations](https://github.com/ToriML/onnx2pytorch/tree/master/onnx2pytorch/operations) 
if you encounter problems when importing the models.
A list of pretrained models is also available in the [main page](https://github.com/ToriML/onnx2pytorch#usage).

### Data preparation

The module accepts [`HDF5`](https://www.hdfgroup.org/) files as data sources. 
The file should contain the samples as the 
format [NCHV](https://oneapi-src.github.io/oneDNN/dev_guide_understanding_memory_formats.html).

Note that, while the standardization can be performed through the APIs themselves (preferred), the preprocessing 
such as resize, reshape, rotation and normalization should be applied in this step.

An example, that creates a subset of the imagenet dataset, can be found in [this gist](https://gist.github.com/maurapintor/25a6d80f9f86d36f72a4b2cc8540008f).

### How to start a security evaluation job
A security evaluation job can be enqueued with a `POST` request to `/api/security_evaluations`. 
The API returns the **job unique ID** that can be used to access **job status** and **results**. 
Running workers will wait for new jobs in the queue and consume them with a FIFO rule.

The request should specify the following parameters in its body:
* **dataset** (*string*): the path where to find the dataset to be loaded (validation dataset should be used, otherwise check out the "indexes" input parameter).
* **trained-model** (*string*): the path of the onnx trained model.
* **performance-metric** (*string*): the performance metric type that should be used to evaluate the system adversarial robustness. Currently implemented only the `classification-accuracy` metric.
* **evaluation-mode** (*string*): one of 'fast', 'complete'. A fast evaluation will perform the experiment with a subset of the whole dataset (100 samples). For more info on the fast evaluation, see [this paper](https://dl.acm.org/doi/10.1145/3310273.3323435).
* **task** (*string*): type of task that the model is supposed to perform. This determines the attack scenario. (available: "classification" - support for more use cases will be provided in the future).
* **perturbation-type** (*string*): type of perturbation to apply (available: "max-norm" or "random").
* **perturbation-values** (*Array of floats*): array of values to use for crafting the adversarial examples. These are specified as percentage of the input range, fixed, in [0, 1] (*e.g.*, a value of 0.05 will apply a perturbation of maximum 5% of the input scale).
* **indexes** (*Array of ints*): if the list of indexes is specified, it will be used for creating a specific sample from the dataset.
* **preprocessing** (*dict*): dictionary with keys "mean" and "std" for defining custom preprocessing. The values should be expressed as lists. If not set, standard imagenet preprocessing will be applied. Otherwise, specify an empty dict for no preprocessing.

```json
{
  "dataset": "<dataset-path>.hdf5",
  "trained-model": "<model_path>.onnx",
  "performance-metric": "classification-accuracy",
  "evaluation-mode": "fast",
  "task": "classification",
  "perturbation-type": "max-norm",
  "perturbation-values": [
    0, 0.01, 0.02, 0.03, 0.04, 0.05
  ]
}

```

The API can also be tested with Postman (it is configured already to get the ID and use it for fetching results):

[![Run in Postman](https://run.pstmn.io/button.svg)](https://god.gw.postman.com/run-collection/1276122-97709dd2-5b99-4737-ae94-2c9868b776f4?action=collection%2Ffork&collection-url=entityId%3D1276122-97709dd2-5b99-4737-ae94-2c9868b776f4%26entityType%3Dcollection%26workspaceId%3D9c875dc5-2201-4035-a06d-6567bd8a75e6)

### Job status API
Job status can be retrieved by sending a `GET` request to `/api/security_evaluations/{id}`, where the 
id of the job should be replaced with the job ID of the previous point. A `GET` to `/api/security_evaluations` 
will return the status of all jobs found in the queues and in the finished job registries. 

### Job results API
Job results can be retrieved, once the job has entered the `finished` state, with a `GET` request 
to `/api/security_evaluations/{id}/output`. A request to this path with a job ID that is not yet 
in the `finished` status will redirect to the job status API.

## Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. 
Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. 
You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

1. Fork the Project
2.  Create your Feature Branch (git checkout -b feature/AmazingFeature)
3.  Commit your Changes (git commit -m 'Add some AmazingFeature')
4.  Push to the Branch (git push origin feature/AmazingFeature)
5.  Open a Pull Request

If you don't have time to contribute yourself, feel free to open an issue with your suggestions.

## License 
TODO


## Credits

Based on the [Security evaluation module](https://gitlab.com/aloha.eu/security_evaluation) - 
[ALOHA.eu project](http://aloha-h2020.eu)
