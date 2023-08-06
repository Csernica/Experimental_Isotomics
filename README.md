# Experimental_Isotomics

This repository consists of code used to process the data and generate the figures in "High-Dimensional Isotomics, Part 2: Observations of Over 100 Constraints on Methionine’s Isotome"

First, the "read Process CSV" directory contains code to read in the directly observed constraints (readMN) and use these to reconstruct isotopologue-specific enrichment (processMN) from .csv files obtained by applying FTStatistic to .RAW files generated from Orbitrap-IRMS measurements. These .RAW and .csv files are publicly available at a repository cited in the paper. 

It also includes the output of both readMN and processMN, as .json files.

Next, "Isotomics Scripts" includes several .py files associated with "High-Dimensional Isotomics, Part 1" which are used extensively in the subsequent data processing. 

Finally, the figure specific directories contain a .py file used to generate the figures in the text of the manuscript. In some cases, figures generated via these .py files were further processed to create the figures. 
