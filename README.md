# Leveraging Tabular Transformer Architectures for Operating System Fingerprinting

arXiv Preprint

**Authors:** 
[Rubén Pérez-Jove](https://linktr.ee/rubenpjove), [Alejandro Pazos](https://orcid.org/0000-0003-2324-238X), [Jose Vázquez-Naya](https://orcid.org/0000-0002-6194-5329)

## Description
This project focuses on Operating System fingerprinting using [Tabular Transformers](https://github.com/lucidrains/tab-transformer-pytorch) with different datasets. The aim is to explore the application of advanced deep learning architectures in the field of network security. The repositoy is structured in different folders, according to the experiments executed on the three different datasets used.


## Installation
The project was developed using Python 3.10.8. To install the necessary dependencies, run the following command. 

```sh
pip install -r requirements.txt
```

## Datasets
The datasets used in this project are not uploaded to this repository due to GitHub file size restrictions. They can be downloaded from the original sources, listed below:

- **`DAT1`**: [lastovicka_2023_passiveOSRevisited](https://zenodo.org/doi/10.5281/zenodo.7635137)
- **`DAT2`**: [lastovicka_2019_usingTLS](https://zenodo.org/doi/10.5281/zenodo.3461770)
- **`DAT3`**: [nmap-7.94_2023_OSdb](https://svn.nmap.org/nmap-releases/nmap-7.94/)

## Results
The results of the experiments are documented in the Results Section. This includes performance metrics, visualizations, and analysis.

## Contributing
Contributions are welcome! Please contact the main author or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Acknowledgements
Special thanks to my supervisors, Alejandro Pazos and Jose Vázquez-Naya, and the [RNASA-IMEDIR research group](https://investigacion.udc.es/es/Research/Details/G000282) for their support.