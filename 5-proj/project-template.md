---
title: "Project Title"
subtitle: "Subtitle"
author: "Author Name"
date: "DD Month, YYYY"
output:
  pdf_document:
    toc: true
    toc_depth: 4
    fig_caption: true
fontsize: 12pt
geometry: "left=1cm,right=1cm,top=1.5cm,bottom=1.5cm"
header-includes:
  - \usepackage[section]{placeins}
  - \usepackage{fixltx2e}
  - \usepackage{longtable}
  - \usepackage{pdflscape}
  - \usepackage{graphicx}
  - \usepackage{caption}
  - \usepackage{gensymb}
  - \usepackage{subcaption}
  - \DeclareUnicodeCharacter{2264}{$\pm$}
  - \DeclareUnicodeCharacter{2265}{$\geq$}
  - \usepackage{fancyhdr}
  - \usepackage{lipsum}
---

```{.python .pandoc-pyplot caption="Python Setup" output="setup.png" include=FALSE}
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set universal theme for figures
plt.style.use('seaborn')

# Configure plot settings
plt.rcParams['figure.figsize'] = (6, 4.5)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.grid'] = True
# Set numerical precision
np.set_printoptions(precision=5)
```

Links: [Google](http://www.google.com)
Image: ![image](logo.gif)
Bold Text: **Ram**
Italic Text:  _Krishna_
Code Chunk:
```python
import os
```
# Guides for Writing your report

For the full document:

* you can use a html output but you need to keep the sectioning
* You are writing a comprehensive report
* The goal is to communicate to others
* Need to focus on communicating
* There should be no code or text not in the appropriate headings below
* Delete/comment out extra notes (like these) or others so your document is as clear as it can be
* Reference your figures, tables, equations in the document.
* Captions are needed on all figures, tables, and equations.
* Think about your audience and clearly communicating.

For the updates:

* please include the previous update for each Submitted Update Document
    + Each update is important to keep for grading
* For the final report remove updates and give a full comprehensive report
* Delete or comment out my notes that you are not using when you submit your documents
    + The extra clutter makes communication difficult


# Update 2

* Please put a bulleted list of things you have accomplished since the last update
  + Include things that didn't work but you tried
  + Things you are planning on doing
  + Questions that you might have on your project.
* Reference the sections and figures you are dicussing here

# Update 1

We have decided to use the dataset QM24, we are able to manage its data and to print in 3D the center of the atoms composing the molecules.

* Please put a bulleted list of things you have accomplished since the last update
  + Include things that didn't work but you tried
  + Things you are planning on doing
  + Questions that you might have on your project.
* Reference the sections and figures you are dicussing here

# Excuetive Summary
  
* Summarize the key (This could be a bulleted list)
  + information about your data set
  + major data cleaning
  + findings from EDA
  + Model output
  + Overall conclusions

# Abstract

* Summary of the nature, finding and meaning of your data analysis project. 
* 1 paragraph written summary of your data analysis project
   
# Introduction

* Background and motivation of the Data Science question. The ``Why'' of the research
* Explanation of your data
  + Where is your data from
  + What are the variables
* What data would be necessary to improve your analysis?
   
# Data Science Methods

We decided to realize a Property prediction of the elements from their 3D molecular structure. We will use Supervising trqining on Neural Network. To do so we will use the 3D properties of the molecules as training inputs, then we will train the network with the use of the desired Property (Atomization Energy for example) as a label.

The input data should need no pretreatment. The output label should be a continous value defining the property of the element.

The main limit of this method is for each property we would lik to predict, we would have to entirely redo the training with a different label.  

# Exploratory Data Analysis

## Explanation of your data set

* How many variables?
* What are the data classes?
* How many levels of factors for factor variables?
* Is your data suitable for a project analysis?
* Write you databook, defining variables, units and structures

## Data Cleaning

* What you had to do to clean your data

## Data Vizualizations

* Vizualizations of your data

## Variable Correlations

* Pairwise correlation plots, etc.
  
# Statistical Learning: Modeling \& Prediction

*  least 1 simple linear model (or simple logistic model)
* requires the appropriate modeling for your data set including machine learning

* Types of modeling to try
* Statistical prediction/modeling
* Model selection
* Cross-validation, Predictive R2
* Interpret results
* Challenge results
   
# Discussion

* Discussion of the answers to the data science questions framed in the introduction
  
# Conclusions
   
# Acknowledgments
   
# References

* Include a bib file in the markdown report
* Or hand written citations.