# HEAT: Hyperbolic Embedding of Attributed Networks
Reference implementation of algorithm described in https://arxiv.org/abs/1903.03036

Authors: David MCDOANLD dxm237@cs.bham.ac.uk

## Requirements:
Python3
Numpy
Scipy
Scikit-learn
Scikit-multilearn
Keras


## How to use:
TODO

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
@misc{mcdonald2019heat,
    title={HEAT: Hyperbolic Embedding of Attributed Networks},
    author={David McDonald and Shan He},
    year={2019},
    eprint={1903.03036},
    archivePrefix={arXiv},
    primaryClass={cs.SI}
}
```