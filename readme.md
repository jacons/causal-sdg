# Tabular Data Generator with Structural Causal Models

![Dataset cover](imgs/dataset-cover.png)

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python Version](https://img.shields.io/badge/Python-%3E=3.11-blue.svg)](https://www.python.org/downloads/)

## Description

This framework provides a tool for generating synthetic tabular data based on Structural Causal Models (SCMs).
By utilizing SCMs, the framework allows for the explicit definition of causal relationships between variables,
ensuring that the generated data reflects these underlying dependencies. This approach is particularly useful for:

* **Realistic data generation:** Creating datasets that preserve the complex causal interactions present in real-world
  data.
* **Data augmentation:** Increasing the size and diversity of existing datasets while maintaining causal consistency.
* **Model testing and validation:** Generating controlled data to evaluate the behavior of machine learning algorithms
  in different scenarios.
* **Simulations:** Conducting "what-if" experiments and analyzing the consequences of interventions on variables.
* **Privacy-preserving data sharing:** Sharing synthetic data that retains important statistical characteristics without
  revealing sensitive information.

The framework offers a flexible interface for defining causal graphs, specifying the functions that describe the
relationships between variables, and generating datasets of arbitrary sizes.

|    | Occupation shared among the datasets                                              | Top-5 skills                                                                                                                                                                           |
|---:|:----------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | Sales workers                                                                     | ['customer service', 'organising, planning and scheduling work and activities', 'quality assurance procedures', 'wholesale and retail sales', 'provide high quality customer service'] |
|  1 | Teaching professionals                                                            | ['computer programming', 'computer graphics', 'project management', 'teach teaching principles', 'web programming']                                                                    |
|  2 | Legal, social and cultural professionals                                          | ['office software', 'project management', 'business ICT systems', 'computer programming', 'conduct scholarly research']                                                                |
|  3 | Personal care workers                                                             | ['work under supervision in care', 'geriatrics', 'nursing and midwifery', 'computer programming', 'project management']                                                                |
|  4 | Handicraft and printing workers                                                   | ['use IT tools', 'electronics', 'Adobe Illustrator', 'computer graphics', 'use technical documentation']                                                                               |
|  5 | Food processing, wood working, garment and other craft and related trades workers | ['customer service', 'provide high quality customer service', 'project management', 'office software', 'web programming']                                                              |
|  6 | Numerical and material recording clerks                                           | ['office software', 'customer service', 'business ICT systems', 'manage administrative systems', 'project management']                                                                 |
|  7 | Hospitality, retail and other services managers                                   | ['wholesale and retail sales', 'customer service', 'project management', 'management skills', 'sales strategies']                                                                      |
|  8 | Science and engineering professionals                                             | ['computer programming', 'computer graphics', 'project management', 'office software', 'electronics and automation']                                                                   |
|  9 | Food preparation assistants                                                       | ['use cooking techniques', 'adapt props', 'secretarial and office work', 'pallets loading', 'planning and organising']                                                                 |
| 10 | Building and related trades workers, excluding electricians                       | ['repair plumbing systems', 'use genre painting techniques', 'construction methods', 'carpentry', 'authenticate documents']                                                            |
| 11 | Business and administration professionals                                         | ['office software', 'business ICT systems', 'wholesale and retail sales', 'project management', 'accounting']                                                                          |
| 12 | Production and specialised services managers                                      | ['project management', 'business ICT systems', 'office software', 'management skills', 'logistics']                                                                                    |
| 13 | Labourers in mining, construction, manufacturing and transport                    | ['prepare orders', 'operate forklift', 'pick orders for dispatching', 'logistics', 'computer programming']                                                                             |
| 14 | Information and communications technology professionals                           | ['computer programming', 'web programming', 'business ICT systems', 'operating systems', 'Java (computer programming)']                                                                |
| 15 | Agricultural, forestry and fishery labourers                                      | ['office software', 'joining parts using soldering, welding or brazing techniques', 'hotel, restaurants and catering', 'turf management', 'perform tungsten inert gas welding']        |
| 16 | Commissioned armed forces officers                                                | ['search engines', 'driving light vehicles', 'supervision of persons', 'management skills', 'project management']                                                                      |
| 17 | Other clerical support workers                                                    | ['office software', 'business ICT systems', 'human resource management', 'business and administration', 'project management']                                                          |
| 18 | Personal service workers                                                          | ['office software', 'project management', 'hair removal techniques', 'hotel operations', 'customer service']                                                                           |
| 19 | Science and engineering associate professionals                                   | ['computer programming', 'computer graphics', 'electricity', 'project management', 'office software']                                                                                  |
| 20 | Cleaners and helpers                                                              | ['clean equipment', 'office software', 'provide domestic care', 'operate recycling processing equipment', 'sales strategies']                                                          |
| 21 | Chief executives, senior officials and legislators                                | ['business and administration', 'computer programming', 'project management', 'office software', 'data mining']                                                                        |

## Example dataset

An example of dataset generated with this framework is available on Kaggle:
[Synthetic Job Offer and Curricula](https://www.kaggle.com/datasets/jaconsx/synthetic-job-offer-and-curricula).

## Installation

```bash
    conda env create --file reqs.yml
    conda activate CausalSDG
```

## Authors

* [Andrea Iommi] ([Jacons](https://github.com/jacons))

