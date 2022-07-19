from datetime import date

from owlready2 import *


class OntologyInstanceGenerator:

    def __init__(self, vehicle, hsn, tsn, dtc, ontology_path, ontology_file):
        self.vehicle = vehicle
        self.hsn = hsn
        self.tsn = tsn
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

        file = "ontology_instance_{}_{}_{}.owl".format(self.hsn, self.tsn, date.today())
        self.onto.save(file)

    def add_dtc(self):
        dtc = self.onto.DTC(self.dtc)
        self.onto.DTC(dtc)

    def add_fault_condition(self):
        # TODO: retrieve from DB
        fault_condition = "dummy_fault_cond"
        c = self.onto.FaultCondition(fault_condition)
        self.onto[self.dtc].represents.append(c)

    def add_fault_causes(self):
        # TODO: retrieve from DB
        fault_causes = ["causeOne", "causeTwo", "causeThree"]
        for fault in fault_causes:
            cause = self.onto.FaultCause(fault)
            fault_condition = list(self.onto[self.dtc].represents)[0]
            fault_condition.hasCause.append(cause)

    def add_fault_symptoms(self):
        # TODO: retrieve from DB
        symptoms = ["sympOne", "sympTwo", "sympThree"]
        for symptom in symptoms:
            s = self.onto.Symptom(symptom)
            fault_condition = list(self.onto[self.dtc].represents)[0]
            fault_condition.manifestedBy.append(s)

    def add_suspect_component(self):
        # TODO: retrieve from DB
        sus_components = ["susOne", "susTwo", "susThree"]
        for sus in sus_components:
            comp = self.onto.SuspectComponent(sus)
            self.onto[self.dtc].pointsTo.append(comp)

    def add_occurs_with(self):
        # TODO: retrieve from DB
        dtc_occurring_with = ["PXXXX", "PYYYY"]
        for dtc in dtc_occurring_with:
            code = self.onto.DTC(dtc)
            self.onto[self.dtc].occursWith.append(code)

    def add_fault_category(self):
        # TODO: retrieve from DB
        fault_cat = "category A"
        cat = self.onto.FaultCategory(fault_cat)
        self.onto[self.dtc].hasCategory.append(cat)

    def add_fault_description(self):
        # TODO: retrieve from DB
        fault_desc = "This is fault X - test - test - test"
        desc = self.onto.FaultDescription(fault_desc)
        self.onto[self.dtc].hasDescription.append(desc)

    def add_measuring_positions(self):
        # TODO: retrieve from DB
        measuring_pos = ["pos_A", "pos_B", "pos_C"]
        for pos in measuring_pos:
            measuring_position = self.onto.MeasuringPos(pos)
            self.onto[self.dtc].implies.append(measuring_position)

    def add_corrective_actions(self):
        # TODO: retrieve from DB
        corrective_actions = ["perform test A", "check sensor B", "apply C"]
        dtc = self.onto[self.dtc]
        fault_condition = list(self.onto[self.dtc].represents)[0]
        for act in corrective_actions:
            action = self.onto.CorrectiveAction(act)
            action.deletes.append(dtc)
            action.resolves.append(fault_condition)
            self.onto.CorrectiveAction(action)

    def add_vehicle(self):
        fault_condition = list(self.onto[self.dtc].represents)[0]
        vehicle = self.onto.Vehicle(self.vehicle)
        vehicle.HSN.append(self.hsn)
        vehicle.TSN.append(self.tsn)
        fault_condition.occurredIn.append(vehicle)


if __name__ == '__main__':
    instance_gen = OntologyInstanceGenerator(
        "Mazda 3", "849357984", "453948539", "P1111", "../OBDOntology", "obd_ontology.owl"
    )
    instance_gen.create_ontology_instance()
