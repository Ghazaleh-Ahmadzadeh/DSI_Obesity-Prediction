# **Classifying Obesity Levels Using Dietary and Lifestyle Information from Individuals in Colombia, Peru, and Mexico**

Data Science/ Machine Learning Software Foundations Certificate Program, Data Sciences Institute, University of Toronto

Cohort 6 - Team ML #6

This project focuses on an in-depth analysis of the “Estimation of Obesity Levels Based on Eating Habits and Physical Condition” dataset using Machine Learning models to determine which dietary and lifestyle attributes the most significant predictors of obesity levels. Our proposal outlines the development of the "Compass Proactive Health Platform" to identify individuals at risk of developing severe obesity, thereby enabling targeted preventative measures to reduce associated medical healthcare costs.

## **Contents**
* [Team Members](#team-members)
* [Introduction](#introduction)
* [Objectives](#objectives)
* [Methodology](#methodology)
* [Key Findings](#key-findings)
* [Conclusion](#conclusion)
* [Folder Structure](#folder-structure)
* [How to use this repo](#how-to-use-this-repo)
  
## **Team Members**

* Maria Rossano ([rossanot](https://github.com/rossanot))  
* Reshma Rajendran ([EzhavaReshma](https://github.com/EzhavaReshma))
* Ghazaleh Ahmadzadeh ([Ghazaleh-Ahmadzadeh](https://github.com/Ghazaleh-Ahmadzadeh))  
* Melanie Cheung See Kit ([melcsk](https://github.com/melcsk))  
* Elizabeta Radaeva ([eradaeva1](https://github.com/eradaeva1))  
* Cristian Nicolas Cordova Puglianini ([NicoForce](https://github.com/NicoForce))

## **Introduction**
Obesity is a diagnosis given to individuals with excessive body fat and calculated Body Mass Index (BMI) of greater or equal to 30kg/m<sup>2</sup>. It often becomes a long-term and chronic health condition that is associated with increased risks of other complications, such as type 2 diabetes, heart disease, and cancer. Thus, patients’ treatments have become a heavy burden to the healthcare system. In the Americas region, obesity is a prevalent condition among adults (Fig. 1) and is estimated to cost 985.99 billion USD, mostly in medical expenses ([Okunogbe et al., 2022, e009773](https://pubmed.ncbi.nlm.nih.gov/36130777/)). 

![Fig1](docs/figures/fig1-obesity.png)
Fig 1. % of Adults with obesity (BMI  ≥ 30kg/m2) across selected countries (Data Tables | World Obesity Federation Global Obesity Observatory, 2025).

In this context, developing preventative measures to address obesity must be considered paramount. As a result, people’s quality of life would improve, thereby relieving the strain on the healthcare system, especially for countries where some form of universal health coverage is provided by the government. For example, in Colombia, about 19% of government spending was directed towards healthcare, representing approximately 6.6% of Colombia’s Gross Domestic Product (GDP) in 2021. ([Health in the Americas, Pan American Health Organization: Colombia Profile](https://hia.paho.org/en/country-profiles/colombia))

### **Business Motivation**
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

From exploratory data analysis, it was found that the target classes are not heavely imbalanced (see [Figure 2](#fig2)), as a result, no bias is expected from this component of the data.

![Fig2](docs/figures/fig2-target-classes.png)
Fig 2. Distribution of classes of the target


The linear correlation between the various numerical features in the dataset was evaluated showing that no strong dependency between features existed (see [Figure 3](#fig3)). 

![Fig3](docs/figures/fig3-corr.png)
Fig 3. Linear correlation between numerical features

#### **SMOTE**
Note that for this dataset, authors generated 77% of the data synthetically using Weka tool and SMOTE filter while 23% was collected directly from participants via a web platform.

## **Objectives**

## **Methodology**
Exploratory data analysis (EDA), data preprocessing and model training experiment outputs are stored in the `data` folder using the corresponding timestamp of the experiment execution time.

In all cases, Jupyter notebooks were used to test code before implementing it into the final pipeline. 

### About project management 
- The team held regular virtual stand ups to discuss the project objectives, progress, concerns, and . 
- A google docs document was employed to keep track of the project progress, references and other documentation, model experiments planning, team ideation/brain stormming, meetings agenda and outcomes

### About project versioning using Git/GitHub
- Team members open and merged at least one pull request (PR) 
- At least two team members were assigned to reviewing each PR
- The team member to open the PR was not necesarilly the person to close it
- Branches were <ins>deleted</ins> after revision and merging were completed

### **a. Data extraction**
Raw data was directly obtained as a .csv file from the database [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) and stored in the `data/raw` folder for preprocessing.

### **b. Exploratory data analysis (EDA)**
EDA was performed on the raw and preprocessed datasets. It consisted of gaining fundamental insights of the datatasets such as the number of examples (rows) and features (columns), the type of data, missing values, duplicate examples, outliers, feature value distribution.

For the above we employed measures of central tendency, as well as a series of histograms, bar plots, and correlation matrices.

The EDA pipeline outputs, a report in .txt format and plots in .png format, are stored in the [data/eda](./data/eda/) folder.

### **c. Data preprocessing**
To prevent bias during model training, the features `weight` and `height` were scaled using the minmax scaler, within the range (1, 5), as implemented in scikit-learn to match the ranges of the rest of the numerical features.

Categorical variables , `Gender`, `family_history_with_overweight`, `FAVC`, `CAEC`, `SMOKE`, `SCC`, `CALC`, `MTRANS`, including the target `NObeyesdad` where encoded accordingly. 

### **d. Model training**
Team members tried different models and tools, see Table 2. 

#### **Table 2.** Experiments model details
| **Team member** | **Maria** | **Melanie** | **Ghazaleh** | **Cristian** |
|---|---|---|---|---|
| **Algorithm** | Random Forest Classifier | Random Forest Classifier |     [LightGBM](https://lightgbm.readthedocs.io/en/stable/) Classifier |     XGBClassifier |
| **Best parameters** | `{'criterion':   'entropy', 'max_depth': 15, 'n_estimators': 68}` | `{'max_depth': 21,   'n_estimators': 450}` | `{'learning_rate':   0.05, 'max_depth': 7, 'n_estimators': 350, 'num_leaves': 20}` | `{'classifier__learning_rate': 0.05, 'classifier__max_depth': 5,   'classifier__n_estimators': 325, 'classifier__subsample': 1.0}` |
| **Hyperparameter optimization   method** | [Optuna](https://optuna.org/) | [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) | GridSearchCV | GridSearchCV |
| **CV** | Yes | Yes | Yes | Yes |
| **Other** | Feature selection using   `mutual_info_classif` |  |  |  |
| **Performance** | CV Accuracy: 0.947 | Train Accuracy: 0.953  | Train Accuracy: 1.0 | Train Accuracy: 0.972 |
|  | Test Accuracy: 0.950 | Test Accuracy: 0.939 | Test Accuracy: 0.969 | Test Accuracy: 0.967    |

For the final implementation, the best performing models from [Table 2](#table-2-experiments-model-details) were chosen, using Optuna for hyperparameter optimization, cross-validation, and feature selection. In addition, a baseline using a decision tree with default parameters was included. 

### **e. Model validation**
### **f. Feature analysis**
### **g. Dashboard for model showcase**

## **Key Findings**
### About the dataset:
- No outliers were found
- Most of the numerical features show a Gaussian distribution
- Target classes are 

### About model training


## **Conclusion**

## **Folder Structure**

```bash
.
├───data
│   ├───eda
│   ├───preprocessed
│   └───raw
├───docs
│   └───figures
├───experiments
│   ├───model-exp-team-member.py
│   └───README.md
├───eda
│   └───eda.py
├───models
├───notebooks
├───README.md
```

- **data**: Contains raw and processed datasets
  - **eda**: EDA pipeline outputs, including a report .txt file and plots
  - **preprocessed**: preprocessed datasets
  - **raw**: source raw data
- **docs**: figures and tool files
 - **figures**: documentation figures and plots
- **eda**: EDA pipeline
- **models**: Contains files for model training and testing, as well as the finalized model
- **notebooks**: pre-implemented pipelines
- **README.md**: This file

## **How to use this repo**
It is recommended to create a dedicated environment for this repo,

```bash
conda env create -f environment.yml
```

If not installed,

```bash
pip install -r requirements.txt
```

Each component of the pipeline, for example data processing and model training, can be run using the `run.sh` file found in the project main directory as shown below.

- EDA:

```bash
bash run.sh eda
```

- Data processing

```bash
bash run.sh preproc
```

- Model training
- Model validation
- Dashboard
