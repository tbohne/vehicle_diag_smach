FROM python:3.10-buster

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

COPY ./high_level_smach.py /code/high_level_smach.py

COPY ./config.py /code/config.py

RUN pip install -r /code/requirements.txt

CMD ["python3", "high_level_smach.py"]
