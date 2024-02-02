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

python evaulate.py
=======
python evaluate.py



