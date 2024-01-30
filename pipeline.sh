# 1. Inversion
cd /home/oskar/PycharmProjects/DataAssimilation/Inversion/
igm_run

# 2. Merge Input
cd /home/oskar/PycharmProjects/DataAssimilation/Preprocessing/
python create_input_nc.py

# 3. Reference Run
cd /home/oskar/PycharmProjects/DataAssimilation/ReferenceRun/
igm_run

# 4. Data Assimilation
cd /home/oskar/PycharmProjects/DataAssimilation
python data_assimilation.py

python evaluate.py


