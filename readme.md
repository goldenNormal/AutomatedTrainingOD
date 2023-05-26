# EntropyStop
The official code for paper "Unleashing the Potential of Unsupervised Deep Outlier Detection through Automated Training Stopping".

## Dependencies
This code requires the following:
- Python>=3.7
- PyTorch>=1.12.1
- Numpy>=1.19.2
- Scipy>=1.6.2
- Scikit-learn>=0.24.1
- PyG  >= 2.1.0
- Pyod

## Generate dataset
For image dataset of MNIST-3 and MNIST-5, run the following cmd to generate the dataset.

```
cd data
```

```
python3 generate_MNIST.py
```


## Baselines: Naive, Optimal, Entropy
To get the Naive, Optimal, Entropy results of AE:
```
cd ae
```
```
python3 run_ae_naive.py
```
```
python3 run_ae_optimal.py
```
```
python3 run_ae_entropy.py
```

To get the Naive, Optimal, Entropy results of RDP:
```
cd rdp
```
```
python3 run_rdp_naive.py
```
```
python3 run_rdp_optimal.py
```
```
python3 run_rdp_entropy.py
```

To get the Naive, Optimal, Entropy results of DeepSVDD:
```
cd svdd
```
```
python3 run_svdd_mode.py --train naive
```
```
python3 run_svdd_mode.py --train optimal
```
```
python3 run_svdd_mode.py --train entropy
```


## Baseline: UOMS

**After running the Naive training of all deep OD algorithms**, run the following cmd to get the result  of UOMS:

```
cd eval_uoms
```

```
python3 run_uoms.py
```

## Baselines: ROBOD, IF 
To run ROBOD,
```
cd ROBOD
```
```
python3 run_ROBOD.py
```

To run IF,
```
cd IF
```
```
python3 run_if.py
```


## Results of AE for injected outliers 
```
cd ae
```

```
python3 run_ae_inject.py
```

## Visualization of loss entropy curve
```
python3 run_ae_single_visual.py
```

```
python3 run_rdp_single_visual.py
```
The HP configuration can be tuned to see different visualizations. The  figures are stored  in the path of './training-img/'.
