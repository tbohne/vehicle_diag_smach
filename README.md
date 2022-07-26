# AW4.0 Prototype

State-machine-based prototype of a vehicle diagnosis *[and recommendation]* system.

## Dependencies

- [**standalone-smach**](https://pypi.org/project/standalone-smach/): ROS SMACH fork for development of HSM outside of ROS
- [**AW_40_GUI**](https://github.com/DanielNowak98/AW_40_GUI):  GUI for parsing OBD data and vehicle meta information
- [**BeautifulSoup**](https://pypi.org/project/beautifulsoup4/): library that makes it easy to scrape information from web pages
- [**OBDOntology**](https://github.com/tbohne/OBDOntology): ontology for capturing knowledge about on-board diagnostics (OBD), particularly diagnostic trouble codes (DTCs) + ontology query tool

## State Machine Architecture

<object data="https://github.com/tbohne/AW4.0_Prototype" type="application/pdf" width="700px" height="700px">
    <embed src="img/smach_v8.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/tbohne/AW4.0_Prototype/img/smach_v8.pdf">Download PDF</a>.</p>
    </embed>
</object>

