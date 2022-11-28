# Vehicle Diagnosis State Machine

State-machine-based prototype of a vehicle diagnosis *[and recommendation]* system.

## Dependencies

- [**standalone-smach**](https://pypi.org/project/standalone-smach/): ROS SMACH fork for development of HSM outside of ROS
- [**BeautifulSoup**](https://pypi.org/project/beautifulsoup4/): library that makes it easy to scrape information from web pages
- [**tensorflow**](https://pypi.org/project/tensorflow/): open source machine learning framework
- [**OBDOntology**](https://github.com/tbohne/OBDOntology): ontology for capturing knowledge about on-board diagnostics (OBD), particularly diagnostic trouble codes (DTCs) + ontology query tool
- [**oscillogram_classification**](https://github.com/tbohne/oscillogram_classification): neural network based anomaly detection for vehicle components using oscilloscope recordings
- [**CustomerXPS**](https://github.com/tbohne/CustomerXPS): expert system that deals with customer complaints
- [**py4j**](https://www.py4j.org/): bridge between Python and Java
- [**Apache Jena Fuseki**](https://jena.apache.org/documentation/fuseki2/): SPARQL server hosting / maintaining the knowledge graph
- [**networkx**](https://pypi.org/project/networkx/): Python package for creating and manipulating graphs and networks

## Installation

```
$ mkdir diag
$ cd diag
$ git clone https://github.com/tbohne/vehicle_diag_smach.git
$ git clone https://github.com/tbohne/OBDOntology.git
$ git clone https://github.com/tbohne/oscillogram_classification.git
$ git clone https://github.com/tbohne/CustomerXPS.git
$ touch __init__.py setup.py
```
Open `setup.py` and enter the following:
```python
from setuptools import setup, find_packages
setup(
    name='diag',
    packages=find_packages(),
    package_data={'': ['img/*.ico', 'img/*.png', '*.owl', 'res/*']},
    include_package_data=True,
)
```
Install:
```
$ pip install . --user
```
Additionally, to use the customer XPS, create a .jar file with the py4j and d3web dependencies (cf. [CustomerXPS](https://github.com/tbohne/CustomerXPS)).

## Usage

Run server (knowledge graph) from Apache Jena Fuseki root directory (runs at `localhost:3030`) - for configuration cf. [OBDOntology](https://github.com/tbohne/OBDOntology):
```
$ ./fuseki-server
```
Run customer XPS server (in `CustomerXPS/`):
```
$ java -jar out/artifacts/CustomerXPS_jar/CustomerXPS.jar
```
Run state machine (in `diag/`):
```
$ python vehicle_diag_smach/high_level_smach.py
```

## State Machine Architecture
![](img/smach_v11.jpg)

## Fault Isolation Result Example
![](img/isolation.png)
