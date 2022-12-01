FROM ubuntu:20.04

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

COPY ./vehicle_diag_smach/ /code/vehicle_diag_smach/

COPY ./res/ /code/res/

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip install --upgrade pip
RUN apt-get install -y git
RUN pip install Cython

RUN pip install -r /code/requirements.txt

ENV TF_CPP_MIN_LOG_LEVEL=3

CMD ["python3", "/code/vehicle_diag_smach/high_level_smach.py"]
