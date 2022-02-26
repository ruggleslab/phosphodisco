# PhosphoDisco (PhDc)
A python package for finding co-regulated phospho sites from phosphoproteomics and proteomics data.   


## Description
PhDc uses optimized clustering to find co-regulated phosphosite modules. Then, PhDc uses protein and phosphoprotein abundance data from kinases and phosphatases to nominate module regulators. It can also use sample annotation data to identify modules significantly correlated with clinical variables. 

## Tutorial
You can find the [tutorial jupyter notebook](tutorial/phosphodisco_tutorial.ipynb) in the tutorial folder.
To get started, follow these steps:
1. Clone the repo:
`git clone git@github.com:ruggleslab/phosphodisco.git`
2. Set up the environment needed to run the notebook. We will use [mamba](https://mamba.readthedocs.io/en/latest/) for this, which is a drop-in replacement for conda:
```
conda install -c conda-forge mamba
mamba env create -f phosphodisco/env.yml -p phdc_env
```
3. Activate the environment and install phosphodisco
```
conda activate phdc_env/
cd phosphodisco
python setup.py develop
```
4. Start jupyter lab and check out the notebook:
```
cd tutorial
jupyter lab
```
The tutorial will run you through the major steps in running the phosphodisco pipeline.


### PhDc can perform pre-processing steps:
1. Use regularized linear regression to normalize phosphopeptide data by protein data. 
2. Column/sample normalize data.  
    a. Median normalization.  
    b. Upper quartile normalization.  
    c. Two-component median normalization.   

#### NO Z-SCORING

##### Finding modules:
For either mode 1 or 2, PhDc can also be used to find reproducibly clustered sites.   
**TODO: describe hypercluster a little**

### Following clustering, PhDc can be used for these analyses:
1. Nominate regulators: given protein and phosphoprotein abundances of kinases and phosphatases, PhDc can find the most correlated putative regulators for each module. 
2. Correlate module scores with clinical continuous and categorical features. Identify modules most high correlated with each feature.  

## Input data
n$^|$ = # p-sites in original table  
m = # samples  
k = # different sample annotations  

1. Phosphosite x Sample table of log2 relative phospho abundances (e.g. from iTRAQ). csv or tsv. n$^|$*m
2. Sample x annotation table for clinical annotations. csv or tsv. m*k
3. Continuous/categorical labels for annotation table. csv or tsv. k\*1  
## Outputs
n = # p-sites after normalization  
m = # samples  
i = # different optimization conditions  
j = # modules in best clustering  
l = # putative regulators, i.e. kinases and phosphatases
1. Protein-normalized phospho data (optional, only if normalizing) n\*m
2. Phosphosite x clustering attempts table of p-site module labels. tsv. n\*i (optional, only if optimizing)  
3. Best scoring clustering result table of p-site module labels. tsv. n\*1\
    a. Heatmaps of p-sites x samples with annotations for each module  
    b. Module scores x samples. tsv. j\*m
        i. Clustered heatmap of module scores vs samples
    c. Coefficients of each kinase and phosphatase x modules. tsv. l\*j    
        i. Clustered heatmap of high scoring regulators vs modules.  
4. Reproducible clusters, phosphosites x cluster labels n\*\1. (optional, only if using optimized clustering)

## Package structure
```
phosphodisco
|_parsers.py
|_classes.py
|_analyze_clusters.py
|_visualize.py
|_cli.py
|_tests
    |_test_parsers.py
    |_test_analyze_clusters.py
    |_test_visualize.py
    |_test_cli.py
data
|_kinase_list.tst
|_phosphatase_list.txt
|_all_list.txt
|_acetylase_list.txt
|_deacetylase_list.txt

docs
|_docs.rst
|_conf.py
|_requirements.txt
LICENSE
README.md
setup.py
```
