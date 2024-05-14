## PriCE: Privacy-Preserving and Cost-Effective Scheduling for Parallelizing the Large Medical Image Processing Workflow over Hybrid Clouds 


Running deep neural networks for large medical images is a resource-hungry and time-consuming task with centralized computing. While outsourcing such medical image processing tasks to hybrid clouds has benefits, such as a significant reduction of execution time and monetary cost, due to privacy concerns, it is still challenging to process sensitive medical images over clouds, which would hinder their deployment in many real-world applications. To overcome this, we first formulate the overall optimization objectives of the privacy-preserving distributed system model, i.e., minimizing the amount of information about the private data learned by the adversaries throughout the process, reducing the maximum execution time and cost under the user budget constraint. We propose a novel privacy-preserving and cost-effective solution called PriCE to solve this multi-objective optimization problem. We performed extensive simulation experiments for artifact detection tasks on medical images using an ensemble of five deep convolutional neural network inferences as the workflow task. Experimental results show that PriCE successfully splits a wide range of input gigapixel medical images with graph-coloring-based strategies, yielding desired output utility and lowering the privacy risk, maximum completion time, and monetary cost under the maximum total cost. 

#### Download data
Download the data from `https://surfdrive.surf.nl/files/index.php/apps/files/?dir=/Research%20Datasets/WSI_dataset&fileid=14843054309`

and place it at `/root/PriCE/dataset/1WSI/data/`

#### Environment setup

Use the terminal for the following steps:

1. Create the environment from the `environment.yml` file:

    ```conda env create -f environment.yml```
2. Activate the new environment: ```conda activate myenv```

3. Verify that the new environment was installed correctly:

    ```conda env list```

#### Folder explanation
1. dataset: storing the datasets, e.g., the original WSI example and its intermediate data files, etc. 
2. inference: storing the CNN inference models. 
3. pipeline-example for artifact detection: storing the application using CNN inference models for artifact detection in WSI. 
4. PriCE-exps: storing the experimental worklows and /or Jupyter notebooks of the PriCE experiments and simulations
    
    4.1 how to split a gigapixel medical image?
        `PriCE/PriCE-exps/graph_coloring_based_image_splitting.ipynb`
        `PriCE/PriCE-exps/evenly_split_w_wo_shuffle.ipynb`

    4.2 how to encrypt/decrypt sensitive information of medical images? How to quantify the privacy-preserving goals?
        `PriCE/PriCE-exps/pertubedata_privacy_risk_evaluation.ipynb` (data perturbation and its privacy-preserving algorithm evaluation)

    
    4.3 how to seek the 3D Pareto optimal resource planning?
        `PriCE/PriCE-exps/Pareto_3D_evaluation.ipynb`
