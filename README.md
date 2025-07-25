# Classifying Obesity Levels Using Dietary and Lifestyle Information from Individuals in Colombia, Peru, and Mexico

Data Science/ Machine Learning Software Foundations Certificate Program, Data Sciences Institute, University of Toronto

Cohort 6 - Team ML #6 Project

This project focuses on an in-depth analysis of the “Estimation of Obesity Levels Based on Eating Habits and Physical Condition” dataset using Machine Learning models to determine which dietary and lifestyle attributes the most significant predictors of obesity levels. Our proposal outlines the development of the "Compass Proactive Health Platform" to identify individuals at risk of developing severe obesity, thereby enabling targeted preventative measures to reduce associated medical healthcare costs.

## Contents
* [Team Members](#team-members)
* [Introduction](#introduction)
* [Objectives](#objectives)
* [Methodology](#methodology)
* [Key Findings](#key-findings)
* [Conclusion](#conclusion)
* [Folder Structure](#folder-structure)
* [How-to use this repo](#how-to)
  
## Team Members

* Maria Rossano ([rossanot](https://github.com/rossanot))  
* Reshma Rajendran Shivdas Kunjpilla ([EzhavaReshma](https://github.com/EzhavaReshma))
* Ghazaleh Ahmadzadeh ([Ghazaleh-Ahmadzadeh](https://github.com/Ghazaleh-Ahmadzadeh))  
* Melanie Cheung See Kit ([melcsk](https://github.com/melcsk))  
* Elizabeta Radaeva ([eradaeva1](https://github.com/eradaeva1))  
* Cristian Cordova ([NicoForce](https://github.com/NicoForce))

## Introduction
Obesity is a diagnosis given to individuals with excessive body fat and calculated Body Mass Index (BMI) of greater or equal to 30kg/m<sup>2</sup>. It often becomes a long-term and chronic health condition that is associated with increased risks of other complications, such as type 2 diabetes, heart disease, and cancer. Thus, patients’ treatments have become a heavy burden to the healthcare system. In the Americas region, obesity is a prevalent condition among adults (Fig. 1) and is estimated to cost 985.99 billion USD, mostly in medical expenses ([Okunogbe et al., 2022, e009773](https://pubmed.ncbi.nlm.nih.gov/36130777/)). 

![Fig1](docs/figures/fig1-obesity.png)
Fig 1. % of Adults with obesity (BMI  ≥ 30kg/m2) across selected countries (Data Tables | World Obesity Federation Global Obesity Observatory, 2025).

In this context, developing preventative measures to address obesity must be considered paramount. As a result, people’s quality of life would improve, thereby relieving the strain on the healthcare system, especially for countries where some form of universal health coverage is provided by the government. For example, in Colombia, about 19% of government spending was directed towards healthcare, representing approximately 6.6% of Colombia’s Gross Domestic Product (GDP) in 2021. ([Health in the Americas, Pan American Health Organization: Colombia Profile](https://hia.paho.org/en/country-profiles/colombia))


### Business Motivation
Here, we propose to determine the features that have the most meaningful impact on their obesity status. These factors range from an individual’s medical history, dietary and health habits to fitness activity. We aim to achieve this goal by training a machine learning model within the context of a classification problem. By identifying the dietary and lifestyle factors influencing obesity in individuals, healthcare providers could help improve tailored solutions for patients that could be translated into a higher treatment success rate. 

- **Client**: Goverment within a context of B2G scheme
- **End User**: Health providers, e.g., hospitals and physicians

### **Dataset Details**
* Source: [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) 
* Sample size: 2111
* Features: 16
* Target variable: Obesity level
* Target labels:
Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, and Obesity Type III

| Feature Name        | Type | Description |
| ----------------- | ------ | ----------------- |
| Gender      | Categorical      | Categorical variable indicating the biological sex of the individual              |
| Age     | Continuous      | Numerical variable representing the age of the individual in years              |
| Height     | Continuous      | Numerical variable representing the height of the individual. Used to calculate BMI and other health indicators.              |
| Weight      | Continuous      | Numerical variable representing the body weight of the individual. Combined with height to assess obesity levels              |
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

## Methodology

## Key Findings

## Conclusion

## Folder Structure

```bash
.
├───data
│   ├───eda
│   ├───preprocessed
│   └───raw
├───docs
│   └───figures
├───experiments
├───models
├───notebooks
├───README.md
```

- **data**: Contains raw and processed datasets
  - **eda**: EDA pipeline
  - **preprocessed**: preprocessed datasets
  - **raw**: source raw data
- **docs**: figures and tool files
 - **figures**: documentation figures and plots
- **models**: Contains files for model training and testing, as well as the finalized model
- **notebooks**: pre-implemented pipelines
- **README.md**: This file

## How-to use this repo
There's an environment.yml with the dependencies, if a conda environment doesn't exist yet, it can be created with the needed dependencies with this command:

```
conda env create --file environment.yml
```

Otherwise, if the conda environment name already exists, it can be updated to match the `environment.yml` file with the following command:

```
conda env update --file environment.yml --prune
```
