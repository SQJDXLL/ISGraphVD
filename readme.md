
## How-to-Run

### Generate SSGraph 



To generate SSGraphs, run the following commands under ~/graphConstruct.
If you modify the code in main.scala, you need to use the following instructions. 
```
    cd ~/graphConstruct/standalone-ext
    sbt clean && sbt stage
    cd ../
    ./runStandalone.sh curl CVE-2021-22901
```

If you only replace the vulnerability without making any changes, use the following instructions.
```
    cd ~/graphConstruct
    ./runStandalone.sh curl CVE-2021-22901
```
    
### SSGraph to Matrix
To extract node and edge matrices from SSGraph, run the following command under ~/graphMatrix.
```
    python graph2matrix.py --project curl --cve_id CVE-2021-22901
```

### Model and detect
train 
run the following command under ~/gmnDetect.
The trained model will be saved to ～/data/project/cve/model
```
    python dividedataset.py --project curl --cve_id CVE-2021-22901
    conda activate zyl_new
    cd gmn
    python train_disjoint.py --project curl --cve_id CVE-2021-22901 --gpu 0 --batch_size 32 --learning_rate 1e-4 --num_epoch 10
```
detect
```
    cd ../detector
    python chooseSample.py --project curl --cve_id CVE-2021-22901 --gpu 0 --largeOrnot True
    python find_th.py --project curl --cve_id CVE-2021-22901 --gpu 0 --largeOrnot True
    python detect.py --project curl --cve_id CVE-2021-22901 --gpu 0 --largeOrnot True
```

