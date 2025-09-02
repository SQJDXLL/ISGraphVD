# ISGraphVD

## About
We propose ISGraphVD, a graph-based vulnerability de-
tection approach designed to be robust across different architectures, compilers, compiler versions, and optimization settings. This approach ensures high detection accuracy
while minimizing false alarms 

---

## QuickStart
### Preparation

### 1) Requirements
- OS: **Ubuntu 20.04+**
- Python: **3.9+**
- Environment: Managed via **conda**
- Dependencies: Listed in [`environment.yml`](./environment.yml)
### 2) Installation
```bash
git clone https://github.com/SQJDXLL/ISGraphVD.git
cd ISGraphVD
conda env create -f environment.yml && conda activate gt
```

### 3)How-to-Run

-  Generate ISGraph 

To generate ISGraphs, run the following commands under ~/graphConstruct.
If you modify the code in main.scala, you need to use the following instructions. 
```bash
    cd ~/graphConstruct/standalone-ext
    sbt clean && sbt stage
    cd ../
    ./runStandalone.sh <project_name> <CVE>
```

If you only replace the vulnerability without making any changes, use the following instructions.
```bash
    cd ~/graphConstruct
    ./runStandalone.sh <project_name> <CVE>
```
    
### ISGraph to Matrix
To extract node and edge matrices from ISGraph, run the following command under ~/graphMatrix.
```bash
    python graph2matrix.py --project <project_name> --cve_id <CVE>
```

### Model and detect
train 
run the following command under ~/gmnDetect.
The trained model will be saved to ~/data/project/cve/model
```bash
    python dividedataset.py --project <project_name> --cve_id <CVE>
    cd gmn
    python train_disjoint.py --project <project_name> --cve_id <CVE> --gpu <gpu_num> --batch_size <batch_size> --learning_rate <learning_rate> --num_epoch <num_epoch>
```
detect
```bash
    cd ../detector
    python chooseSample.py --project <project_name>  --cve_id <CVE> --gpu <gpu_num> --largeOrnot True
    python find_th.py --project <project_name>  --cve_id <CVE> --gpu <gpu_num> --largeOrnot True
    python detect.py --project <project_name>  --cve_id <CVE> --gpu <gpu_num> --largeOrnot True
```

