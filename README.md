# KA-GAT

git clone https://github.com/pvishalkaushik/KA-GAT.git      
            
cd KA-GAT         
pyenv install 3.11.0       
pyenv virtualenv 3.11.0 ka_gnn       
pyenv local ka_gnn       
pip install -r requirements.txt      
          
            

python predict.py --task [task name] --smiles [smiles string]      


# Currently, allowed task names are: bbbp, clintox, tox21, sider 


