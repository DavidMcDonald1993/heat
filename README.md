# HEAT: Hyperbolic Embedding of Attributed Networks
Reference implementation of algorithm described in https://link.springer.com/chapter/10.1007/978-3-030-62362-3_4

Authors: David MCDONALD (davemcdonald93@gmail.com) and Shan HE (s.he@cs.bham.ac.uk)

## Requirements:
Python3
Numpy
Scipy
Scikit-learn
Scikit-multilearn
Keras

## Required packages (pip)
Install the required packages with pip install -r requirements.txt 

## Setup environment (conda) 
The conda environment is described in heat_env.yml.

Run 
'''
conda env create -f heat_env.yml
'''
to create the environment, and 
'''
conda activate heat-env
'''
to activate it. 


## How to use:
Run the code with 
'''
python main.py --edgelist path/to/edgelist.tsv --features path/to/features.csv -e num_epochs
'''
Additional options can be viewed with 
'''bash
python main.py --help
'''


## Input Data Format
Graphs are given as edgelists in the form 
u\tv\tw
where w is the weight of the connection between nodes u and v
u and v should be integers
every int in [0, N-1] must appear in the edgelist

labels and features should be comma separated tables indexed by node id

## Citation:
If you find this useful, please use the following citation
```
@inproceedings{mcdonald2020heat,
  title={HEAT: Hyperbolic Embedding of Attributed Networks},
  author={McDonald, David and He, Shan},
  booktitle={International Conference on Intelligent Data Engineering and Automated Learning},
  pages={28--40},
  year={2020},
  organization={Springer}
}

```