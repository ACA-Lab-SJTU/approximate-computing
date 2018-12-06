# approximate-computing
Source code for approximate computing related works, including one DAC paper and two ICCAD papers.

# Invocation-driven Neural Approximate Computing with a Multiclass-Classifier and Multiple Approximators

The original paper here: <https://arxiv.org/abs/1810.08379>

## Abstract

Neural approximate computing gains enormous energy-efficiency at the cost of tolerable quality-loss. A neural approximator can map the input data to output while a classifier determines whether the input data are safe to approximate with quality guarantee. However, existing works cannot maximize the invocation of the approximator, resulting in limited speedup and energy saving. By exploring the mapping space of those target functions, in this paper, we observe a nonuniform distribution of the approximation error incurred by the same approximator. We thus propose a novel approximate computing architecture with a Multiclass-Classifier and Multiple Approximators (MCMA). These approximators have identical network topologies and thus can share the same hardware resource in a neural processing unit(NPU) clip. In the runtime, MCMA can swap in the invoked approximator by merely shipping the synapse weights from the on-chip memory to the buffers near MAC within a cycle. We also propose efficient co-training methods for such MCMA architecture. Experimental results show a more substantial invocation of MCMA as well as the gain of energy-efficiency.

## Dataset

From AxBench: <http://axbench.org/>

# AXNet: ApproXimate computing using an end-to-end trainable neural network

This is the source codes based on TensorFlow of the paper *AXNet: ApproXimate computing using an end-to-end trainable neural network*.

arXiv: https://arxiv.org/abs/1807.10458

## Codes Strcuture

- build.py ---- The codes build the models of AXNet.
- error.py ---- The error metrics
- example.ipynb ---- The example Jupyter Notebook to build, train, evaluate the AXNet.
- multi-task-learning.ipynb ---- The codes to build, train, evaluate the multi-task learning method.
- utilities.py
- data
    - gen_data.py ---- The codes to convert the original \*.fann file to \*.x \*.y files.
    - the data of different benchmarks



## Citation

```
@inproceedings{peng2018axnet,
  title={AXNet: ApproXimate computing using an end-to-end trainable neural network},
  author={Peng, Zhenghao and Chen, Xuyang and Xu, Chengwen and Jing, Naifeng and Liang, Xiaoyao and Lu, Cewu and Jiang, Li},
  booktitle={Proceedings of the International Conference on Computer-Aided Design},
  pages={11},
  year={2018},
  organization={ACM}
}
```

# On Quality Trade-off Control for Approximate Computing Using Iterative Training

Original paper: 
https://dl.acm.org/citation.cfm?id=3062294

