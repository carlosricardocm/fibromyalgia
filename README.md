# Associative Memories Experiments
This repository contains the data and procedures to replicate the expermients presented in the paper 

Application of Support Vector Machines in the Diagnosis and Treatment of Fibromyalgia


The code was written in Python 3 and was run on a desktop computer with the following specifications:
* CPU: Intel Core i7-6700 at 3.40 GHz
* GPU: Nvidia GeForce GTX 3070
* OS: Windows 11
* RAM: 64GB

### Requeriments
The following libraries need to be installed beforehand:

* numpy
* scikit-learn

The experiments were run using the Anaconda 3 distribution.

### Data
The datasets containing anonymous data of patients with fibromyalgia were provided in CSV format and were used for all the experiments conducted.

### Use

To run the experiment for dataset 1 run:

```shell
python fibro-socio-cuadro.py
```

To run the experiment for dataset 2 run:
```shell
python python fibro-socio-cuadro-maltrato.py
```

To run the experiment for dataset 3 run:

```shell
python python fibro-socio-cuadro-maltrato-escala.py
```

## License

Copyright [2024] Quetzal Natalia Galán López and Carlos Ricardo Cruz Mendoza

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
