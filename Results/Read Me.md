# Project Overview



This project integrates Molecular Dynamics (MD) simulations and Machine Learning (ML) to predict the potential energy of Cu–Ni alloy systems across varying:

* Temperatures: 300–1500 K
* Cu compositions: 10%, 30%, 50%, 70%
* Simulation sizes: 10 Å, 20 Å



MD simulations were conducted using LAMMPS with a Cu–Ni EAM alloy potential. The resulting dataset was used to train ML regression model to efficiently estimate potential energy without running expensive simulations.



### Repository Structure





##### CuNi-PotentialEnergy-ML/

│

├── MD Simulation/Lammps code folder/

│   ├── in.ml\_data\_generator\_cuni         # LAMMPS Code for data generation

│   ├── CuNi.eam.alloy                    # EAM Potential

│   ├── ML\_CuNi\_Training\_Data.txt         # Training Data 

│

├── ML Model/

│   ├── ml\_analysis\_cuni.py   # ML Model Code      

│

├── References/

│   ├── All the references

│

├── Report/

│   ├── Report

│   

├── Results/

│   ├── Graphs             

│   └── Evaluation metric calculated values         

│

└── README.md





### Project Workflow



A) Dataset Generation — LAMMPS



* A parameterized LAMMPS script (in.tensile\_cuni) is automatically filled using Python.



* 40 combinations are generated based on:

1. Temperature
2. Cu composition
3. Simulation size



* For each simulation:

1. Structure is created
2. Equilibrated (NVT + NPT)
3. Tensile deformation is applied
4. Potential energy and stress/strain are extracted



* All results are saved into CSV files.



B) ML Model Training

The pipeline includes:



* Data preprocessing



* Train-test split



* Model training:

1. Random Forest



* Cross-validation (5-fold)



* Model metrics:

1. R²
2. MAE
3. RMSE



### Technologies Used

* In Molecular Dynamics Part

1. LAMMPS



* In Machine Learning Part

1. Python
2. NumPy, Pandas
3. Scikit-learn
4. Matplotlib, Seaborn



### Usage



1\. Generate LAMMPS Inputs

2\. Run All LAMMPS Simulations

3\. Train the Machine Learning Model

4\. Make New Predictions





### Results Summary



* ML predicts Cu–Ni potential energy with enough accuracy.
* Cross-validation confirms strong generalization.
* Random Forest performed properly and predicted correct.





























