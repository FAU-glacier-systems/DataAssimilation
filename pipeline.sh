# 1. Inversion
cd /home/oskar/PycharmProjects/DataAssimilation/data_inversion/v2
igm_run

# 2. Merge Input
cd /home/oskar/PycharmProjects/DataAssimilation/preprocessing/
python create_input_nc.py

# 3. Reference Run
cd /home/oskar/PycharmProjects/DataAssimilation/reference_run/v2
igm_run

# 4. Data Assimilation
cd /home/oskar/PycharmProjects/DataAssimilation
python data_assimilation.py



