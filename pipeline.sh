# 1. Inversion
cd Inversion/
igm_run
cd ..

# 2. Merge Input
cd Preprocessing/
python create_input_nc.py
cd ..

# 3. Reference Run
cd ReferenceRun/
igm_run
cd..

# 4. Data Assimilation
python data_assimilation.py

<<<<<<< HEAD
python evaulate.py
=======
python evaluate.py
>>>>>>> a99125e0ac905f5a2970ef3ff423a4663d0f6e82


