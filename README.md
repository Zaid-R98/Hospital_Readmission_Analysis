
# About this Project

This project aims to predict whether a certain patient is more likely to be readmitted to a hospital based on several factors.

The factors used to predict whether a patient would be readmitted or not are listed below: 

 *1. Age
 2. Time Spent in Hospital
 3. Number of Medications
 4. Number of diagnosis
 5. Metformin level
 6. Chlorpropamide level
 7. Glimepiride level
 8. Tolazamide level
 9. Insulin level
 10. Race
 11. Admission Type
 12. Admission Source*
 
 The details of the models and the data pre-processing and cleaning done to the dataset can be found in detail in the **.ipynb** file.

## How to run the web app:

The model once trained, was used by a web app we built using Python's flask framework.

To run the Web Application, follow these steps:

 1. Install Python on your local machine
 2. Create and activate a Python virtual enviroment
 3. Run the following command: `pip install -r requirements.txt`

 4. Once the dependancies are installed, run the following command:`flask run`

