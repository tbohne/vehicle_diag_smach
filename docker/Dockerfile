FROM ubuntu:20.04

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./vehicle_diag_smach/ /code/vehicle_diag_smach/
COPY ./setup.py /code/setup.py
COPY ./res/ /code/res/
COPY ./docker/run_services.sh /code/run_services.sh

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip install --upgrade pip
RUN apt-get install -y git
RUN apt-get install -y jq
RUN apt-get install -y curl
RUN apt-get install -y tmux

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y libx11-dev
RUN apt-get install -y python3-tk

RUN pip install Cython
RUN pip install -r /code/requirements.txt
RUN pip install /code/

# silence tensorflow logs
ENV TF_CPP_MIN_LOG_LEVEL=3

# install OpenJDK
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/
RUN export JAVA_HOME

ARG VERSION=4.9.0
ENV JENA_URL=https://dlcdn.apache.org/jena/binaries/apache-jena-fuseki-${VERSION}.tar.gz
RUN curl -L $JENA_URL > /code/apache-jena-fuseki.tar.gz
RUN tar -xvzf /code/apache-jena-fuseki.tar.gz
RUN chmod +x /code/apache-jena-fuseki-${VERSION}/fuseki-server
ENV FUSEKI_BASE=/code/apache-jena-fuseki-${VERSION}
EXPOSE 3030
COPY ./docker/shiro.ini $FUSEKI_BASE/shiro.ini

RUN chmod +x /code/run_services.sh
ENTRYPOINT ["/code/run_services.sh"]
