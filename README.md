# Classifying Obesity Levels Using Dietary and Lifestyle Information from Individuals in Colombia, Peru, and Mexico

Data Science/ Machine Learning Software Foundations Certificate Program, Data Sciences Institute, University of Toronto

Cohort 6 - Team ML #6 Project

This project focuses on an in-depth analysis of the “Estimation of Obesity Levels Based on Eating Habits and Physical Condition” dataset using Machine Learning models to determine which dietary and lifestyle attributes the most significant predictors of obesity levels.

## Contents
* [Team Members](#team-members)
* [Introduction](#introduction)
* [Objectives](#objectives)
* [Methodology](#methodology)
* [Key Findings](#key-findings)
* [Conclusion](#conclusion)
* [Folder Structure](#folder-structure)
  
## Team Members

* Maria Rossano ([rossanot](https://github.com/rossanot))  
* Reshma Rajendran ([EzhavaReshma](https://github.com/EzhavaReshma))
* Ghazaleh Ahmadzadeh ([Ghazaleh-Ahmadzadeh](https://github.com/Ghazaleh-Ahmadzadeh))  
* Melanie Cheung See Kit ([melcsk](https://github.com/melcsk))  
* Elizabeta Radaeva ([eradaeva1](https://github.com/eradaeva1))  
* Cristian Cordova ([NicoForce](https://github.com/NicoForce))

## Introduction

Dataset Details
* Source: [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) 
* Sample size: 2111
* Features: 16
* Target variable: Obesity level
* Target labels:
Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, and Obesity Type III

| Feature Name        | Type | Description |
| ----------------- | ------ | ----------------- |
| Gender      | Categorical      |               |
| Age     | Continuous      |               |
| Height     | Continuous      |               |
| Weight      | Continuous      |               |
| family_history_with_overweight      | Binary     | Has a family member suffered or suffers from being overweight?              |
| FAVC | Binary | Do you eat high caloric food frequently? |
| FCVC | Integer | Do you usually eat vegetables in your meals? |
| NCP | Continuous | How many main meals do you have daily? |
| CAEC | Categorical | Do you eat any food between meals? |
| SMOKE | Binary | Do you smoke? |
| CH2O | Continuous | How much water do you drink daily? |
| SCC | Binary | Do you monitor the calories you eat daily? |
| FAF | Continuous | How often do you have physical activity? |
| TUE | Integer | How much time do you use technological devices such as cell phone, videogames, television, computer and others? |
| CALC | Categorical | How often do you drink alcohol?|
| MTRANS | Categorical | Which transportation do you usually use? |

Note that for this dataset, authors generated 77% of the data synthetically using Weka tool and SMOTE filter while 23% was collected directly from participants via a web platform.

## Objectives
### Business Motivation
* Stakeholders: Insurance companies  
  - Identify predisposition for complications related to obesity in order to accurately calculate insurance premiums to charge to customer

* Stakeholders: Pharmaceutical company market pitch for GLP-1 agonists (e.g. Wegovy, Ozempic, etc …)
  - Identify emerging markets where need for anti-obesity drugs are currently unmet
 
* Stakeholder: Company marketing health tracker app
  - Connect with personal health tracker app to predict possible complications from current diet and fitness information


## Methodology

## Key Findings

## Conclusion

## Folder Structure
```markdown
.
├── data
├── models
└── README.md
```

 * **data**: Contains raw and processed datasets.
 * **models**: Contains files for model testing and finalized model.
 * **README.md**: This file.
