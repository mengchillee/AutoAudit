# AutoAudit: Mining Accounting and Time-Evolving Graphs

------------

Lee, MC., Zhao, Y., Wang, A., Liang, P.J., Akoglu, L., Tseng, V.S., and Faloutsos, C., AUTOAUDIT: Mining Accounting and Time-Evolving Graphs. **Under submission at *IEEE International Conference on Big Data (IEEE BigData)***, 2020.

##  Introduction
In this paper we propose AutoAudit, a systematic method for handling anomaly detection problems not only in accounting datasets, but also in other real-world datasets. 
It consists four major components:
- **"Smurfing" Detection**: We proposeAA-SMURF, an un-supervised and parameter-free algorithm to detect injected“Smurfing” pattern in real-world datasets.
- **Attention Routing**: We proposeAA-ARto attend to themost suspicious periods in time-evolving graphs and pro-vide explanations.
- **Discoveries**:  We  discover  three  month-pairs  with  highcorrelation, proved by “success stories”, and patterns ofaccounting datasets follow Power Laws in log-log scales.
- **Generality**: We further generalized our method on otherreal-world graph datasets, such as Enron Email and CzechFinancial datasets.

**Let us add a figure here to describe the system:**

In addition to xxxx


## Installation and Dependency
The experiment code is writen in Python 3 and built on a number of Python packages:
- matplotlib==2.0.2
- pandas==0.21.0
- scipy==0.19.1
- numpy==1.13.1
- scikit_learn==0.19.1

Batch installation is possible using the supplied "requirements.txt":

````cmd
pip install -r requirements.txt
````

------------


## Datasets
Three datasets are used (see dataset folder):

| Datasets         | Nodes       | Edges        | Time Span             |
| ---------------- | ----------- | ------------ | ------------------- |
| Accounting       | 254         | 285,298      | 01/01/2016 to 02/06/2017           |
| Czech Financial  | 11,374      | 273,508      | 01/05/1993 to 12/14/1998         |
| Enron Email      | 16,771      | 1,487,863    | 01/01/2001 to 12/31/2001          |

- Czech Financial dataset can be found in https://data.world/lpetrocelli/czech-financial-dataset-real-anonymized-transactions.
- Enron Email dataset can be found in https://www.cs.cmu.edu/~./enron/.

## Usage and Sample Output (Demo Version)
Experiments could be reproduced by running **demo_XXX.py** directly. 
You could simply download/clone the entire repository and execute the code by 

```cmd
python AA-Smurf.py
python AA-AR.py
```

## Results

**We also need to present some initial results**

## Conclusions
In this work, we present AutoAudit, which addresses the anomaly detection problem on time-evolving accounting datasets. This kind of data is usually complicated and hard to organize. Our main purpose is to automatically spot anomalies, such as money laundering, providing huge convenience for auditors and risk management professionals. Our approach is also general enough to be easily modified to solve problems in different domains.
