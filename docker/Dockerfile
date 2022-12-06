FROM ubuntu:20.04

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./vehicle_diag_smach/ /code/vehicle_diag_smach/
COPY ./res/ /code/res/
COPY ./docker/run_services.sh /code/run_services.sh

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip install --upgrade pip
RUN apt-get install -y git
RUN apt-get install -y jq
RUN apt-get install -y curl

RUN pip install Cython
RUN pip install -r /code/requirements.txt

# silence tensorflow logs
ENV TF_CPP_MIN_LOG_LEVEL=3

ENV DEBIAN_FRONTEND=noninteractive

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

# APACHE JENA FUSEKI
ARG KNOWLEDGE_GRAPH_URL=https://raw.githubusercontent.com/tbohne/OBDOntology/main/knowledge_base/obd_knowledge_graph.owl
ENV KNOWLEDGE_GRAPH_FILE=/code/obd_knowledge_graph.owl
RUN curl -L $KNOWLEDGE_GRAPH_URL > $KNOWLEDGE_GRAPH_FILE

ARG VERSION=4.6.1
ENV JENA_URL=https://dlcdn.apache.org/jena/binaries/apache-jena-fuseki-${VERSION}.tar.gz
RUN curl -L $JENA_URL > /code/apache-jena-fuseki.tar.gz
RUN tar -xvzf /code/apache-jena-fuseki.tar.gz
RUN chmod +x /code/apache-jena-fuseki-${VERSION}/fuseki-server
ENV FUSEKI_BASE=/code/apache-jena-fuseki-${VERSION}
EXPOSE 3030
COPY ./docker/shiro.ini $FUSEKI_BASE/shiro.ini

RUN chmod +x /code/run_services.sh
ENTRYPOINT ["/code/run_services.sh"]
