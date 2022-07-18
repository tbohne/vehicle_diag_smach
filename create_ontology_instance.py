from owlready2 import *
from datetime import date
import sys
from os import path

sys.path.append(path.abspath('../OBDOntology'))

VEHICLE = "Mazda 3"
HSN = "849357984"
TSN = "453948539"
DTC = "P1111"


def add_dtc():
    dtc = onto.DTC(DTC)
    onto.DTC(dtc)


def add_fault_condition():
    # TODO: retrieve from DB
    fault_condition = "dummy_fault_cond"
    c = onto.FaultCondition(fault_condition)
    onto[DTC].represents.append(c)


def add_fault_causes():
    # TODO: retrieve from DB
    fault_causes = ["causeOne", "causeTwo", "causeThree"]
    for fault in fault_causes:
        cause = onto.FaultCause(fault)
        fault_condition = list(onto[DTC].represents)[0]
        fault_condition.hasCause.append(cause)


def add_fault_symptoms():
    # TODO: retrieve from DB
    symptoms = ["sympOne", "sympTwo", "sympThree"]
    for symptom in symptoms:
        s = onto.Symptom(symptom)
        fault_condition = list(onto[DTC].represents)[0]
        fault_condition.manifestedBy.append(s)


def add_suspect_component():
    # TODO: retrieve from DB
    sus_components = ["susOne", "susTwo", "susThree"]
    for sus in sus_components:
        comp = onto.SuspectComponent(sus)
        onto[DTC].pointsTo.append(comp)


def add_occurs_with():
    # TODO: retrieve from DB
    dtc_occurring_with = ["PXXXX", "PYYYY"]
    for dtc in dtc_occurring_with:
        code = onto.DTC(dtc)
        onto[DTC].occursWith.append(code)


def add_fault_category():
    # TODO: retrieve from DB
    fault_cat = "category A"
    cat = onto.FaultCategory(fault_cat)
    onto[DTC].hasCategory.append(cat)


def add_fault_description():
    # TODO: retrieve from DB
    fault_desc = "This is fault X - test - test - test"
    desc = onto.FaultDescription(fault_desc)
    onto[DTC].hasDescription.append(desc)


def add_measuring_positions():
    # TODO: retrieve from DB
    measuring_pos = ["pos_A", "pos_B", "pos_C"]
    for pos in measuring_pos:
        measuring_position = onto.MeasuringPos(pos)
        onto[DTC].implies.append(measuring_position)


def add_corrective_actions():
    # TODO: retrieve from DB
    corrective_actions = ["perform test A", "check sensor B", "apply C"]

    dtc = onto[DTC]
    fault_condition = list(onto[DTC].represents)[0]

    for act in corrective_actions:
        action = onto.CorrectiveAction(act)
        action.deletes.append(dtc)
        action.resolves.append(fault_condition)
        onto.CorrectiveAction(action)


def add_vehicle():
    fault_condition = list(onto[DTC].represents)[0]
    vehicle = onto.Vehicle(VEHICLE)
    vehicle.HSN.append(HSN)
    vehicle.TSN.append(TSN)
    fault_condition.occurredIn.append(vehicle)


if __name__ == '__main__':

    onto_path.append("../OBDOntology")
    onto = get_ontology("obd_ontology.owl")
    onto.load()

    add_dtc()
    add_fault_condition()
    add_vehicle()
    add_fault_causes()
    add_fault_symptoms()
    add_suspect_component()
    add_occurs_with()
    add_fault_description()
    add_fault_category()
    add_measuring_positions()
    add_corrective_actions()

    with onto:
        try:
            sync_reasoner(infer_property_values=True)
        except owlready2.base.OwlReadyInconsistentOntologyError:
            print("### reasoner determined inconsistency ###")

    file = "ontology_instance_{}_{}_{}.owl".format(HSN, TSN, date.today())
    onto.save(file)
