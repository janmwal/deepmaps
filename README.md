
# Welcome to deepmaps 
**Deepmaps** is a parameter suggestion pipeline for parkinsonian deep brain stimulation. I've working on since early 2022. It began as my master project at the [ARTORG research center](https://www.artorg.unibe.ch/) and extended later on in 2023. The latest version of this project focuses on ProjNet, the 2D plane projections model. More info in the original thesis abstract below.

## Thesis abstract 
**Background**: Subthalamic Nucleus Deep Brain Stimulation is an effective therapeutic tool to alleviate motor symptoms in Parkinson’s Disease. With the recent innovation of directional stimulation leads, the programming of optimal stimulation settings is becoming a complex and time-consuming challenge. We propose here a data processing pipeline from volumes of tissue activated as well as three distinct deep learning models to predict optimal stimulation settings.

**Methods**: Pre- and post-operative brain imaging from 46 Parkinson’s Disease patients in Inselspital, Bern was used to reconstruct the stimulation leads and generate volumes of tissue activated. After data augmentation and pre-processing, the volumetric (VTANet), multi-view (ProjNet) and shape descriptor (PropsNet) datasets were fed into their respective deep learning models. Each model was tuned with state-of-the-art hyperparameter optimization techniques.

**Results**: ProjNet predicted best level and best contact of a stimulation lead with 97% and 94% accuracy, respectively. Its mean effect threshold error was 0.08 mA and 0.02 mA, respectively. VTANet had 76% accuracy for best level and 56% accuracy for best contact, averaging 0.68 mA for effect threshold error in both cases. PropsNet results were not significant.

**Limitations**: Only mapping data was used for predictions. Side-effects and chronic Unified Parkinson’s Disease Rating Scale evaluations were not included, limiting this study scope to mapping only. Moreover, all models were trained only on Bern datasets, thus lacking external validation from another center with a different mapping protocol.
Conclusion: We explored 3D deep learning for deep brain stimulation. The best model was ProjNet, a 2D multi-view convolutional neural network. However, extending the predictions to chronic clinic assessments and external validation are required before the deep learning models can be launched as a clinical tool for optimal stimulation settings in Parkinson’s Disease patients.

**Keywords**: deep brain stimulation, parkinson’s disease, deep learning, computer-assisted programming
## Setup
This is a Python 3.10.1 environment that mainly uses [pytorch](https://pytorch.org/), [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/stable/),  and [matplotlib](https://matplotlib.org/). Just run `pip install -r requirements.txt` to use set it up.

## Data

Please note that the lab's data used in this project is not included in this repository. The code provided here is intended for demonstration and educational purposes. To run the pipeline effectively you will need :
- Flipped VTAs into the same hemisphere as nifti files labelled as p{patient_number}_c{contact}_a{stim_amp}.nii (change the name and filepath accordingly in the [dataset-making notebook](notebooks/tables/make_all_table.ipynb)). VTAs should be generated as stated in [this paper](https://doi.org/10.1016/j.brs.2019.05.001). Generating VTAs in the full range 0-5mA with 0.5mA increments is recommended for training sample size, thus better results.
Please note that you will probably need to adapt the notebook quite a bit as the cohort structures are obviously unique to one center.
- a ground-truth csv table containing the centerID, leadID, contactID, relativeImprovement, amplitude, efficiency, and filename. It should contain clinician-verified programming values so the score-inference script can work and the model can use more training samples.


## Use 

1. Once you have VTAs and the table in the format stated above, run [make_all_table.ipynb](notebooks/tables/make_all_table.ipynb) (adapt the filepath if necessary).
It will created complete csv tables in `data/raw/tables`.
2. Run [stn-centered_space_massive.ipynb](notebooks/make_database/stn_algo/stn-centered_space_massive.ipynb) to shift VTAs from MNI template space to the custom STN-centered space.
3. Remove VTA artifacts with [clean_vtas.py](src/data/clean_vtas.py).
4. Add in-between frames (0.1mA increments) with [tweening.py](src/tweening/tweening.py).
5. Generate orthogonal plane projections with [plane_projection.py](src/features/plane_projection.py).
6. Split tuning and training samples with fixed fraction contact sampling in [tune_dataset_split.ipynb](notebooks/tune_dataset_split/tune_dataset_split.ipynb) (SPLIT part)
7. Perform hyperparameter optimization with [tune_proj_net_kfold_test.py](src/models/tune_proj_net_kfold_test.py). Tweak the search space according to your machine capabilities.
8. Run ProjNet on ground-truth samples[predict_loco.py](src/models/predict_loco.py) (leave-one-center-out) and [predict_lopo.py](src/models/predict_lopo.py) (leave-one-patient-out).
9. Evaluate the predictions with [predictions.ipynb](notebooks/predict_eval/predictions.ipynb).



## Repository organization
* [data](./data)
  * [external](./data/external)
  * [raw](./data/raw)
  * [interim](./data/interim)
  * [processed](./data/processed)
* [docs](./docs)
* [logs](./logs)
  - [cleaning](./logs/cleaning)
  - [inclusion](./logs/inclusion)
- [models](./models)
- [notebooks](./notebooks)
  - [data](./notebooks/data)
  - [make_database](./notebooks/make_database)
    - [stn_algo](./notebooks/make_database/stn_algo)
  - [noise](./notebooks/noise)
  - [predict_eval](./notebooks/predict_eval)
    - [plots](./notebooks/predict_eval/plots)
  - [process_dataset](./notebooks/process_datasets)
  - [tables](./notebooks/tables)
    - [plots](./notebooks/tables/plots)
  - [tune_dataset_split](./notebooks/tune_dataset_split)
  - [volumetric_interpolation](./notebooks/volumetric_interpolation)
  - [VTAs](./notebooks/VTAs)
- [src](./src)
  - [data](./src/data)
  - [datasets](./src/datasets)
  - [features](./src/features)
  - [metrics](./src/metrics)
  - [misc](./src/misc)
  - [models](./src/models)
  - [test](./src/test)
  - [tweening](./src/tweening)
  - [util](./src/util)

## Cookiecutter

This project was generated using Cookiecutter. Cookiecutter is a command-line utility that creates projects from cookiecutters (project templates), which are directory templates that include the basic structure and files for a project. It helps to automate the process of setting up a new project by generating the necessary boilerplate code and configuration files based on predefined templates.

For more information about Cookiecutter, visit [Cookiecutter GitHub repository](https://github.com/cookiecutter/cookiecutter).


## License

MIT License

Copyright (c) 2022-2024 Jan Mikolaj Waligorski

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
