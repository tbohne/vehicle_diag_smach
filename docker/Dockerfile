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

ENV TF_CPP_MIN_LOG_LEVEL=3

#############################################
# JAVA

ENV DEBIAN_FRONTEND=noninteractive

# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/
RUN export JAVA_HOME

#############################################

#############################################
# APACHE JENA FUSEKI

# data
ARG TRIPLES_URL=https://raw.githubusercontent.com/tbohne/OBDOntology/main/knowledge_base/obd_knowledge_graph.owl
ENV DATA_FILE=/code/obd_knowledge_graph.owl
RUN curl -L $TRIPLES_URL > $DATA_FILE

ARG VERSION=4.6.1

# jena - build TDB database
# ENV TDB_LOC=/code/fuseki

# ENV JENA_URL=https://repository.apache.org/content/repositories/releases/org/apache/jena/apache-jena/${VERSION}/apache-jena-${VERSION}.tar.gz

ENV JENA_URL=https://dlcdn.apache.org/jena/binaries/apache-jena-fuseki-4.6.1.tar.gz
RUN curl -L $JENA_URL > /code/apache-jena-fuseki.tar.gz
RUN tar -xvzf /code/apache-jena-fuseki.tar.gz

RUN chmod +x /code/apache-jena-fuseki-4.6.1/fuseki-server

# RUN apache-jena-${VERSION}/bin/tdb2.xloader --loc $TDB_LOC $DATA_FILE

# fuseki
# ENV URL=https://repository.apache.org/content/repositories/releases/org/apache/jena/jena-fuseki-server/${VERSION}/jena-fuseki-server-${VERSION}.jar

ENV FUSEKI_BASE=/code/apache-jena-fuseki-4.6.1

COPY ./shiro.ini $FUSEKI_BASE/shiro.ini

# RUN mkdir -p $BASE

# RUN curl --silent --show-error --output fuseki-server.jar $URL

EXPOSE 3030

RUN chmod +x /code/run_services.sh

ENTRYPOINT ["/code/run_services.sh"]
