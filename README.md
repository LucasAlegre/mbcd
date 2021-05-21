# Model-Based RL Context Detection

Code for the paper [Minimum-Delay Adaptation in Non-Stationary Reinforcement Learning via Online High-Confidence Change-Point Detection](https://arxiv.org/abs/2105.09452)

### To reproduce the Half-Cheetah in a Non-Stationary World experiment:

```
cd path/mbcd
pip install -e .
python3 experiments/mbcd_run.py -algo mbcd
```

### Citation:
```
@InProceedings{Alegre+2021aamas,
    title = {Minimum-Delay Adaptation in Non-Stationary Reinforcement Learning via Online High-Confidence Change-Point Detection},
    author = {Lucas N. Alegre and Ana L. C. Bazzan and Bruno C. {\relax da} Silva},
    booktitle = {Proceedings of the 20th International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
    location = {Virtual Event, United Kingdom},
    year = {2021},
    pages = {97--105},
    isbn = {9781450383073},
    publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
    address = {Richland, SC}
}
```
