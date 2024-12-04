# Safety through Feedback in Constrained RL  
**Authors:** Shashank Reddy Chirra, Pradeep Varakantham, Praveen Paruchuri  

## Introduction  
In safety-critical RL, designing cost functions for safe behavior can be complex and costly, for example, in domains like self-driving. We propose a scalable method to infer cost functions from feedback, addressing challenges in long term credit assignment by transforming the task into a supervised classification problem with noisy labels. To minimize feedback collection costs, we introduce a novelty-based sampling mechanism that elicits feedback only novel trajectories. Experiments on Safety Gymnasium and self-driving scenarios demonstrate that our method achieves near-optimal performance, highlighting its effectiveness and scalability.  

## Installation  
1. Install [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).  

2. Clone this repository:  
    ```bash  
    git clone https://github.com/shshnkreddy/RLSF.git  
    cd RLSF  
    ```  

3. Create and activate a conda environment:  
    ```bash  
    conda create -n RLSF python=3.10  
    conda activate RLSF  
    ```  

4. Install dependencies:  
    ```bash  
    git clone https://github.com/PKU-Alignment/safety-gymnasium.git  
    cd safety-gymnasium  
    pip install -e .  
    cd ..  
    pip install -r requirements.txt  
    ```  

## Train RLSF  
To train RLSF, use the following command:  
```bash  
./Scripts/run_train_pref.sh  
```  

## Environments and Hyperparameters  
The following environment specifications are supported:  

1. **Safety Gymnasium**: As defined in the [Safety Gymnasium documentation](https://www.safety-gymnasium.com/en/latest/).  

2. **ICRL Benchmark** (G. Liu et al., 2023): Use `'BiasedPendulum'` and `'BlockedSwimmer'` for the Biased Pendulum and Blocked Swimmer environments, respectively.  

3. **Driver Environments** (D. Sadigh et al., 2017; D. Lindner et al., 2022): Use `'SafeDriverTwoLanes'`, `'SafeDriverBlocked'`, and `'SafeDriverLaneChange'` for the respective driver scenarios.  

The remaining hyperparameters are specified in the `'Parameters'` directory.  

## Credit  
This repository builds upon a fork of [SIM-RL](https://github.com/hmhuy0/SIM-RL.git).  

## Contact  
Email: shashankc@smu.edu.sg  

## Citation  
```  
@inproceedings{  
    chirra2024safety,  
    title={Safety through feedback in Constrained {RL}},  
    author={Shashank Reddy Chirra and Pradeep Varakantham and Praveen Paruchuri},  
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},  
    year={2024},  
    url={https://openreview.net/forum?id=WSsht66fbC}  
}  
```  