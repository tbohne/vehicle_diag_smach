#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from datetime import date

from owlready2 import *


class OntologyInstanceGenerator:

    def __init__(self, vehicle, hsn, tsn, vin, dtc, ontology_path, ontology_file):
        self.vehicle = vehicle
        self.hsn = hsn
        self.tsn = tsn
        self.vin = vin
        self.dtc = dtc

        # load ontology
        onto_path.append(ontology_path)
        self.onto = get_ontology(ontology_file)
        self.onto.load()

    def create_ontology_instance(self):
        self.add_dtc()
        self.add_fault_condition()
        self.add_vehicle()
        self.add_fault_causes()
        self.add_fault_symptoms()
        self.add_suspect_component()
        self.add_occurs_with()
        self.add_fault_description()
        self.add_fault_category()
        self.add_measuring_positions()
        self.add_corrective_actions()

        self.check_consistency_and_save_to_file()

    def check_consistency_and_save_to_file(self):
        with self.onto:
            try:
                sync_reasoner(infer_property_values=True)
            except owlready2.base.OwlReadyInconsistentOntologyError:
                print("### reasoner determined inconsistency ###")

        file = "ontology_instance_{}_{}_{}_{}.owl".format(self.hsn, self.tsn, self.vin, date.today())
        self.onto.save(file)

    def add_dtc(self):
        dtc = self.onto.DTC(self.dtc)
        self.onto.DTC(dtc)

    def add_fault_condition(self):
        # TODO: retrieve from DB
        fault_condition = "Dummy fault condition.."
        fc = self.onto.FaultCondition()
        fc.condition_description.append(fault_condition)
        self.onto[self.dtc].represents.append(fc)

    def add_fault_causes(self):
        # TODO: retrieve from DB
        fault_causes = ["causeOne", "causeTwo", "causeThree"]
        for fault in fault_causes:
            cause = self.onto.FaultCause()
            cause.cause_description.append(fault)
            fault_condition = list(self.onto[self.dtc].represents)[0]
            fault_condition.hasCause.append(cause)

    def add_fault_symptoms(self):
        # TODO: retrieve from DB
        symptoms = ["sympOne", "sympTwo", "sympThree"]
        for symptom in symptoms:
            s = self.onto.Symptom()
            s.symptom_description.append(symptom)
            fault_condition = list(self.onto[self.dtc].represents)[0]
            fault_condition.manifestedBy.append(s)

    def add_suspect_component(self):
        # TODO: retrieve from DB
        sus_components = ["susOne", "susTwo", "susThree"]
        for sus in sus_components:
            comp = self.onto.SuspectComponent()
            comp.component_name.append(sus)
            self.onto[self.dtc].pointsTo.append(comp)

    def add_occurs_with(self):
        # TODO: retrieve from DB
        dtc_occurring_with = ["PXXXX", "PYYYY"]
        for dtc in dtc_occurring_with:
            code = self.onto.DTC(dtc)
            self.onto[self.dtc].occursWith.append(code)

    def add_fault_category(self):
        # TODO: retrieve from DB
        fault_cat = "category_A"
        cat = self.onto.FaultCategory()
        cat.category_name.append(fault_cat)
        self.onto[self.dtc].hasCategory.append(cat)

    def add_fault_description(self):
        # TODO: retrieve from DB
        fault_desc = "This is fault X - test - test - test.."
        desc = self.onto.FaultDescription()
        desc.fault_description.append(fault_desc)
        self.onto[self.dtc].hasDescription.append(desc)

    def add_measuring_positions(self):
        # TODO: retrieve from DB
        measuring_pos = ["pos_A", "pos_B", "pos_C"]
        for pos in measuring_pos:
            measuring_position = self.onto.MeasuringPos()
            measuring_position.position_description.append(pos)
            self.onto[self.dtc].implies.append(measuring_position)

    def add_corrective_actions(self):
        # TODO: retrieve from DB
        corrective_actions = ["perform_test_A", "check_sensor_B", "apply_C"]
        dtc = self.onto[self.dtc]
        fault_condition = list(self.onto[self.dtc].represents)[0]
        for act in corrective_actions:
            action = self.onto.CorrectiveAction()
            action.action_description.append(act)
            action.deletes.append(dtc)
            action.resolves.append(fault_condition)
            self.onto.CorrectiveAction(action)

    def add_vehicle(self):
        fault_condition = list(self.onto[self.dtc].represents)[0]
        vehicle = self.onto.Vehicle()
        vehicle.model.append(self.vehicle)
        vehicle.HSN.append(self.hsn)
        vehicle.TSN.append(self.tsn)
        vehicle.VIN.append(self.vin)
        fault_condition.occurredIn.append(vehicle)


if __name__ == '__main__':
    instance_gen = OntologyInstanceGenerator(
        "Mazda 3", "847984", "45539", "1234567890ABCDEFGHJKLMNPRSTUVWXYZ", "P1111", "../OBDOntology", "obd_ontology.owl"
    )
    instance_gen.create_ontology_instance()
