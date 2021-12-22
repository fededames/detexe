<p align="center" >
<img width="90%" src=/><br />
</p>


# DetEXE
A tool to build and analyze static malware detectors, based on machine learning.
With DetEXE you will be able to evaluate the efficiency of different features extracted from a PE File,
to build a malware detector.

## Installation

To install the latest version:
```
pip install detexe
```

## Setup
1. Create a project layout containing the needed directories to store the data of the project.
```
detexe setup
```

1. Add executable samples to the benign and malware directories. You can obtain them from different sources.
SOREL, ViruSshare... (As you are working with malware samples, please, take the safety measures)
2. Configure the features_selection.txt file with the features you wish to extract from the files.

## How to use
### CLI

1. Train your model.
```
detexe train --model="foo"
```

2. Execute adversarial attacks on your trained model. 

    It is possible to select a specific attack, or all ddiferent attacks with one command:
```
detexe attack padding --model="foo" --malware "/malware/path.exe"
```

```
detexe attack all --model "foo" --malware "/malware/path.exe"
```

3. Compare the trained models.
```
detexe compare 
```
4. Search for optimal parameters to obtain better result in training. This parameteres will be saved in the model directory.
```
detexe tune --model="foo"
```

For more info use the command help.
```
detexe --help
```
### Python
Follow the notebook tutorial.ipynb

## Add your own features
1. Add new feature class in separated file under ./detexe/ped/features/your_feature
2. Update ./features_selection.txt file

## Built With
* [LIEF](https://github.com/lief-project/LIEF) - A cross-platform library which can parse, modify and abstract ELF, PE and MachO formats.
* [EMBER](https://github.com/elastic/ember) - Elastic Malware Benchmark for Empowering Researchers.
* [SecML Malware](https://github.com/pralab/secml_malware) -  Python library for creating adversarial attacks against Windows Malware detectors. 
