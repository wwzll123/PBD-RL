import os
from ruamel.yaml import YAML


# 配置 ruamel.yaml 以保留格式
yaml_loader = YAML()
yaml_loader.preserve_quotes = True
yaml_loader.indent(mapping=2, sequence=4, offset=2)

src_path = r'/root/autodl-tmp/PDB_SingleChain_Target_deduplicated.fasta'
src_yaml_dir = r'/root/autodl-tmp/PXDesign_yaml_single_binder'
tgt_yaml_dir = r'/root/autodl-tmp/PXDesign_yaml_SingleChain_RepTarget'
MSA_dir = r'/root/autodl-tmp/PDB_SingleChain_Rep_Target_MSA'
pkl_dir=tgt_yaml_dir


def has_pipeline_msa(msa_dir_path):
    return all(
        os.path.exists(os.path.join(msa_dir_path, name))
        for name in ('pairing.a3m', 'non_pairing.a3m')
    )

# 确保目标目录存在
os.makedirs(tgt_yaml_dir, exist_ok=True)

# 读取 FASTA 文件，提取目标 PDB ID
target_set = set()
with open(src_path, 'r') as f:
    for line in f:
        if line.startswith('>'):
            pdb_id = line.strip()[1:]   # 去掉 '>'
            target_set.add(pdb_id)

print(f"Loaded {len(target_set)} target IDs from FASTA.")

# 遍历源 YAML 目录
for yaml_file in os.listdir(src_yaml_dir):
    if not (yaml_file.endswith('.yaml') or yaml_file.endswith('.yml')):
        continue

    # 提取 pdb_id_chain
    base_name = os.path.splitext(yaml_file)[0]
    if '_vs_' not in base_name:
        print(f"Skipping {yaml_file}: filename format unexpected.")
        continue
    pdb_id_chain = base_name.split('_vs_')[0]

    if pdb_id_chain not in target_set:
        continue

    # 构造 Pipeline 所需的 MSA 目录路径
    msa_path = os.path.join(MSA_dir, pdb_id_chain)


    # 读取原 YAML
    src_yaml_path = os.path.join(src_yaml_dir, yaml_file)
    with open(src_yaml_path, 'r') as f_in:
        data = yaml_loader.load(f_in)

    # 找到 chains 下的唯一键（链名）
    chains = data.get('target', {}).get('chains', {})
    if not chains:
        print(f"Warning: {yaml_file} has no target.chains section.")
        continue
    
    pdb_id=pdb_id_chain.split('_')[0]
    pkl_path=os.path.join(pkl_dir, f'{pdb_id}.pkl.gz')
    if os.path.exists(pkl_path):
        data['target']['file']=f'{pdb_id}.pkl.gz'

    # 假设只有一个链，获取其名称
    chain_name = next(iter(chains.keys()))
    chain_data = chains[chain_name]

    # 更新或添加 msa 字段
    if has_pipeline_msa(msa_path):
        chain_data['msa'] = msa_path
        print(f"Updated {yaml_file} with msa: {msa_path}")
    else:
        print(f"Warning: Pipeline MSA dir incomplete or missing: {msa_path}")

    # 保存到目标目录
    tgt_yaml_path = os.path.join(tgt_yaml_dir, yaml_file)
    with open(tgt_yaml_path, 'w') as f_out:
        yaml_loader.dump(data, f_out)

    
print("Done.")