import os
import shutil
import sys


src_msa_dir = sys.argv[1]
target_msa_dir = sys.argv[2]
os.makedirs(target_msa_dir, exist_ok=True)

for file in os.listdir(src_msa_dir):
    src_file = os.path.join(src_msa_dir, file)
    if os.path.isdir(src_file):
        continue

    with open(src_file, 'r') as f:
        pdb_id = f.readline().strip()[1:]

    sample_msa_dir = os.path.join(target_msa_dir, pdb_id)
    os.makedirs(sample_msa_dir, exist_ok=True)

    non_pairing_file = os.path.join(sample_msa_dir, 'non_pairing.a3m')
    pairing_file = os.path.join(sample_msa_dir, 'pairing.a3m')
    

    shutil.copy(src_file, non_pairing_file)
    shutil.copy(src_file, pairing_file)
    os.remove(src_file)
    print(f'Prepared MSA dir for {pdb_id}: {sample_msa_dir}')
