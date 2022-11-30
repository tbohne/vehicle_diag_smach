FROM python:3.8.15-buster

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

COPY ./vehicle_diag_smach/ /code/vehicle_diag_smach/

COPY ./res/ /code/res/

RUN pip install -r /code/requirements.txt

ENV TF_CPP_MIN_LOG_LEVEL=3

CMD ["python3", "/code/vehicle_diag_smach/high_level_smach.py"]
