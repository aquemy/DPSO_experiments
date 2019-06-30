# Dependencies

- ```numpy```
- ```scikit-learn```
- ```imbalanced-learn```
- ```PrettyTable```
- ```tqdm```

# Reproducing the experiments

Each individual experiments is described by a scenario written in ```YAML```. To generate all the scenarios ran for the paper, use the following script:

```python scenario_generator.py```

The folder ```./scenario``` should be filled with ```YAML``` files with format ```<algo>_<dataset>_<policy>.yaml```.

Run the experiments using the launcher:

```python experiments_launcher.py```

The launcher is capable distinguish between the scenario already executed, invalid scenarios and scenarios without results. Therefore, even in case of crash, the launcher will start only the remaining scenarios.

The launcher starts the script ```main.py``` for each scenario and output the results in the folder ```./results``` withing three files:

- ```<scenario_name>_<timestamp>.json``` where ```timestamp``` correspond to the moment the scenario finished,
- ```<scenario_name>_stderr.txt```
- ```<scenario_name>_stdout.txt```

The launcher override individual scenario seed attributes. The global seed is defined at the begining of the file:
```GLOBAL_SEED = 42```


# Writing your own scenarios

You can easily write your own based on the following template:

```yaml
title: SVM on Breast with Split policy
setup:
  policy: split
  runtime: 300
  algorithm: SVM
  dataset: breast
control:
  seed: 42
policy:
  step_pipeline: 0
```

The variables of section ```policy``` depends on the policy defined in the ```setup``` section.

# Why is ```hyperopt``` embedded?

The original ```hyperopt``` package provides a stopping criterion in iterations. This can be very biased in practice because configuration evaluation may have different variation for the same algorithm or between several algorithms.

In order to have a timeout at the search level and at each configuration evaluation, we had to modify several core functions of ```hyperopt```. To ensure future compatibility, we embedded the whole repository snapshot at the moment the modifications have been done.