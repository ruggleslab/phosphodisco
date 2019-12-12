# PhosphoDisco (PhDc)

A python package for finding co-regulated phospho sites from phosphoproteomics and proteomics data. 

## Description
PhDc uses optimized clustering to find co-regulated phosphosite modules. Then, PhDc uses protein and phosphoprotein abundance data from kinases and phosphatases to nominate module regulators. It can also use sample annotation data to identify modules significantly correlated with clinical variables. 


### PhDc can perform pre-processing steps:
1. Use regularized linear regression to normalize phosphopeptide data by protein data. 
2. Column/sample normalize data.  
    a. Median normalization.  
    b. Upper quartile normalization.  
    c. Two-component median normalization.   

#### NO Z-SCORING

### PhDc can be run in 3 clustering modes:
1. Full optimization: Compares all clustering algorithms from scikit-learn, and runs through many hyper parameters for each using grid search. 
2. Fast optimization: Compares only fastest algorithms and runs through hyper parameters for those. e.g. k-means. 
3. Defined clustering (fastest): Runs clustering with user-input parameters. 

##### Evaluation
For each mode, models can be chosen in a variety of ways: Either maximizes silhouette score, adj-rand compared to a gold-standard set, or F1 score, or minimizes cost (?). 

##### Alternative clustering output:
For either mode 1 or 2, PhDc can also be used to find reproducibly clustered sites.   
**TODO: is there a way to set thresholds here? found with all parameters? Found with 80% of parameters? How to accumulate clustering results?**

### Following clustering, PhDc can be used for these analyses:
1. Nominate regulators: given protein and phosphoprotein abundances of kinases and phosphatases, PhDc can find the most correlated putative regulators for each module. 
**TODO: use multiple regressions here (linear, SVM, random forest), maybe auto ML? This will be a small problem, so it shouldn't be computationally heavy.**
**TODO: build in dose-response model, which is sigmoid/logistic!**  
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
src
|_autocluster
    |_clustering.py
|_phosreg
    |_parsers.py
    |_classes.py
    |_analyze_clusters.py
    |_visualize.py
|_cli.py
data
|_kinase_list.tst
|_phosphatase_list.txt
|_all_list.txt
|_acetylase_list.txt
|_deacetylase_list.txt
tests
|_test_autocluster
    |_test_clustering.py
|_test_phosreg
    |_test_parsers.py
    |_test_analyze_clusters.py
    |_test_visualize.py
|_test_cli.py
|_test_integration.py
docs
|_docs.rst
|_conf.py
SnakeFile
requirements.txt
LICENSE
README.md
setup.py
```

## SnakeMake integration
There will be a snakemake file for use on SLURM distributed systems that allows the optimized clustering to be distributed to different nodes and run in parallel. It will then collect all the clustering results in a final output.  
The same file can be used to run things in series on a single computer if needed. 
There will also be a function for running everything in series in a snakemake independent way. 

