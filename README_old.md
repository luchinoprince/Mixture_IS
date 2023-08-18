# Robust leave-one-out cross-validation for high-dimensional Bayesian models
Codes for replicating some of the experiments of the paper _Robust leave-one-out cross-validation for high-dimensional Bayesian models_. 
## General information
**PyStan** has undergone a significant upgrade from **PyStan 2** to the newer **PyStan 3** which is not backward compatible in a major way. The current codes where created with **PyStan 2.9.1.1** and hence work only with that version. Also it can have trouble with **Python >=3.9**, hence to run the above code we suggest the following commands to be run in the terminal(assuming one has Anaconda).


    conda create --name pystan python=3.6
    conda activate pystan
    pip install pystan==2.19.1.1 

To run the codes place the [**PSIS**](https://github.com/avehtari/PSIS) and [**ParetoSmooth**](https://github.com/TuringLang/ParetoSmooth.jl) folders in the apporopriate locations. 
## Usage
### Python
- For a pedagogical walk trough of the construction of the estimator read [**tutorial.md**](https://anonymous.4open.science/r/Mixture_IS-A64E/tutorial.md). 

- To test the estimator, together with some of the competing ones in the paper, in a toy-setting where closed form solutions of the quantities are available, one can run [**Gaussian_Simulated.ipynb**](https://anonymous.4open.science/r/Mixture_IS-A64E/Python_codes/Gaussian_simulated.ipynb). Note that in this notebook we leverage the module [**models.py**](https://anonymous.4open.science/r/Mixture_IS-A64E/Python_codes/models.py) which guarantes more robust numerical behaviours and efficiency. To create also the _Pareto-smoothed_ estimator [PSIS](https://github.com/avehtari/PSIS) one must download the repository and place it in a folder accordingly.

- The notebook [Gaussian_Simulated.ipynb](https://anonymous.4open.science/r/Mixture_IS-A64E/Python_codes/Gaussian_simulated_np.ipynb) performs the experiments present in Section 4 of the paper. They are though pretty slow on Python, hence for extensive experiments we have create an apposite **Julia** code(see below). 

### R
- To replicate some of the results on the high-D logistic models in **R** refer to the folder [R_codes](https://anonymous.4open.science/r/Mixture_IS-A64E/R_codes/). For some information regarding the code and the datasets available there look at [**Guide.md**](https://anonymous.4open.science/r/Mixture_IS-A64E/R_codes/guide.md).

### Julia
- For the more computationally heavy experiemnts of Section 4 of the paper we created a julia code which is significantly faster than the Python implementation. The code leverages _multithreading_ and stores the result in a Python-usable format. Hence any type of analysis/visualization of the results can then be performed using the Python notebooks. 
