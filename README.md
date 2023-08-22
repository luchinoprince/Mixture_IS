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

- To test the estimator, together with some of the competing ones in the paper, in a toy-setting where closed form solutions of the quantities are available, one can run [**Gaussian_Simulated.ipynb**](https://anonymous.4open.science/r/Mixture_IS-A64E/Python_codes/Gaussian_simulated.ipynb). Note that in this notebook we leverage the module [**models.py**](https://anonymous.4open.science/r/Mixture_IS-A64E/Python_codes/models.py) which guarantes more robust numerical behaviours and efficiency. To create also the _Pareto-smoothed_ estimator [**PSIS**](https://github.com/avehtari/PSIS) one must download the repository and place it in a folder accordingly.

- The notebook [**Gaussian_Simulated_np.ipynb**](https://anonymous.4open.science/r/Mixture_IS-A64E/Python_codes/Gaussian_simulated_np.ipynb) performs the experiments present in Section 4 of the paper. They are though pretty slow on Python, hence for extensive experiments we have create an apposite **Julia** code(see below). 

### R
- To replicate some of the results on the high-D logistic models in **R** refer to the folder [**R_codes**](https://anonymous.4open.science/r/Mixture_IS-A64E/R_codes/). For some information regarding the code and the datasets available there look at [**Guide.md**](https://anonymous.4open.science/r/Mixture_IS-A64E/R_codes/guide.md).

### Julia
- For the more computationally heavy experiemnts of Section 4 of the paper we created a julia code which is significantly faster than the Python implementation. The code leverages _multithreading_ and stores the result in a Python-usable format. Hence any type of analysis/visualization of the results can then be performed using the Python notebooks. 

## Reproducibility of Figures of manuscript
Not all the figures of the manuscript are directly reproducible from this folder, and not all the figures produced from the current folder are in the manuscript as some of them have more of a pedagocical scope. Here is some general information regarding the latter and some instruction to reproduce some of the former. 

### Python/Julia
- The plot at the end of [**Gaussian_Simulated.ipynb**](https://anonymous.4open.science/r/Mixture_IS-A64E/Python_codes/Gaussian_simulated.ipynb) is just to show the different levels of accuracies of the estimators for the different addends of LOOCV for a single simulated dataset. It has not hence any counterpart in the manuscript. 

- Setting _attempts=10000_ at line _15_ of the third cell block of [**Gaussian_Simulated_np.ipynb**](https://anonymous.4open.science/r/Mixture_IS-A64E/Python_codes/Gaussian_simulated_np.ipynb) and the plotting the results using the penultimate cell block of the same will produce the right subfigure of _Figure 2_ at page _21_ of the manuscript. As mentioned above though, such an experiment is extremely time consuming in Python, hence if one wishes to reproduce it the recommended workflow is the following:
    - Run the Julia code [**hyper_highD.jl**](https://anonymous.4open.science/r/Mixture_IS-A64E/Julia_codes/hyper_highD.jl) with as many threads as physical cores of your working machine; to do so run the following command
        ```
        julia --threads nthreads hyper_highD.jl
        ``` 
        in the terminal substituting _nthreads_ with the number of physical cores of your machine. 
    - Run the first two cell blocks of [**Gaussian_Simulated_np.ipynb**](https://anonymous.4open.science/r/Mixture_IS-A64E/Python_codes/Gaussian_simulated_np.ipynb) and the uncomment the fourth one while skipping the third one. Then normally run the penultimate one to obtain the plots. 
    - To obtain the left subfigure of _Figure 2_ of the manuscript in Python you have to change line _23_ of the third block to 
        ```
        sigma_0 = np.identity(d+1)
        ```
        Nonetheless this has the same computational time issue as before. In Julia you have to change line _176_ of [**hyper_highD.jl**](https://anonymous.4open.science/r/Mixture_IS-A64E/Julia_codes/hyper_highD.jl) to 
        ```
        sigma = Diagonal(1*ones(d+1))
        ```
- The plot at the end of [Leukaemia.ipynb](https://anonymous.4open.science/r/Mixture_IS-A64E/Python_codes/Leukaemia.ipynb) is not present in the paper as here it has more a pedagogical role. We hence opted for a figure which conveyed better the behaviour of infinite variance estimators compared to finite variance ones, while in_ Figure S.4_ of the supplement we chose for an experiment which gave a more quantitative result on the bias of the different estimators. 
- Running the cells of [Stack_Loss.ipynb](https://anonymous.4open.science/r/Mixture_IS-A64E/Python_codes/Stack_Loss.ipynb) directly replicates _Figure S.5_ of the supplement. 

### R
The plot of [**Lppd.R**](https://anonymous.4open.science/r/Mixture_IS-A64E/R_codes/Logistic_Model_R/Lppd.R) just gives an element-wise comparison between the classical and mixture estimators and is not present in the paper, as in the paper we have no plots coming from **R**. It is meant to allow **R** users to get some quick results. The default dataset used is the [**Voice**](https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation) dataset analysed in Section _4.2_ of the manuscript. To analyse the other datasets present in that section change line _16_ of [**Lppd.R**](https://anonymous.4open.science/r/Mixture_IS-A64E/R_codes/Logistic_Model_R/Lppd.R) to the desired dataset. For example if one wished to get the estimators for the [**Parkinson**](https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation) dataset, change line _16_ to 
```
data = read.csv("./../../Data/Parkinson_preprocessed.csv")
```



    

