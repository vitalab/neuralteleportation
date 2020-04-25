---   
<div align="center">    
 
# Neural Teleportation    

<!---
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  

ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)

Conference   
-->   
</div>

 
## Description   

Neural network teleportation using mathematical magic. 

## How to run   
First, install dependencies   
```bash
# clone project
git clone  https://bitbucket.org/vitalab/neuralteleportation.git

# set-up project's environment
cd neuralteleportation
conda env create -f neuralteleportation.yml

# activate the environment so that we install the project's package in it
conda activate neuralteleportation
pip install -e .

```
To test that the project was installed successfully, you can try the following command from the Python REPL:
```python
# now you can do:
from neuralteleportation import Whatever   
``` 


## Known Limitations

* Can't use opperations in the foward method (only nn.Modules)
* Can't reuse modules more than once (Causes error in graph creation and if the layer have teleportation parameters)


### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   