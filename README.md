# CWIDAM
## This repository is a legacy implementation of the paper [Missingness-Pattern-Adaptive Learning With Incomplete Data](https://ieeexplore.ieee.org/iel7/34/4359286/10086606.pdf).

### Requirements:
- Python 3.6.8
- PyTorch 1.0.1

### To produce the experiment results of our method:  
1. in Table 2:
    ```bash
        python linear_model.py
    ```

2. in Table 3:
    ```bash
        python nn_Drive.py
    ```

3. in Table 4:
    ```bash
        python nn_Avila.py
    ```
4. in supplementary material:
    ```bash
        python linear_model_0.py
    ```
    
## Citation
If you find this repository helpful, please consider to cite the following paper:
```
@ARTICLE{10086606,
  author={Gong, Yongshun and Li, Zhibin and Liu, Wei and Lu, Xiankai and Liu, Xinwang and Tsang, Ivor W. and Yin, Yilong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Missingness-Pattern-Adaptive Learning With Incomplete Data}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TPAMI.2023.3262784}}

```
