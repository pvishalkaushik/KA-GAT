# KA-GAT

conda env create -f environment.yml   
conda activate ka_gnn_cpu   
    
python predict.py --task --smiles   
    
Currently, allowed task names are: bbbp, clintox, tox21, sider   
