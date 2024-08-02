#!/bin/bash

# 一键启动s
# chmod +x run.sh


# ./run.sh busybox CVE-2018-20679 1 32 1e-4 20 True True > CVE-2018-20679.log &
# ./run.sh curl CVE-2019-5482 0 32 1e-4 10 True True > CVE-2019-5482.log &
# ./run.sh curl CVE-2021-22901 0 32 1e-4 10 True True > CVE-2021-22901.log &
# tail -f CVE-2021-22901.log

# 查看程序是否还在运行，并且可以看到PID,可以使用kill PID停止进程
# ps aux | grep ./run.sh


PROJECT=$1
CVE_ID=$2
GPU=$3
BATCHSIZE=$4
LR=$5
EPOCH=$6
LARGEORNOT=$7
HL=$8


eval "$(conda shell.bash hook)"
conda activate zyl_new

if [ "$HL" = "True" ]; then
    
    # 添加指令
    cd graphConstruct && ./runStandalone.sh $PROJECT $CVE_ID && cd preprocess && python processGraph.py --project $PROJECT --cve_id $CVE_ID
    cd ../graphHighlight && python makeWinandSlice.py --project $PROJECT --cve_id $CVE_ID
    python diff2Dot.py --project $PROJECT --cve_id $CVE_ID
    python drawHight.py --project $PROJECT --cve_id $CVE_ID && cd ../../graphMatrix
    python highLightGraph2Matrix.py --project $PROJECT --cve_id $CVE_ID --hl && cd ../gmnDetect
    python dividedataset.py --project $PROJECT --cve_id $CVE_ID --hl && cd gmn
    python train_disjoint.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --batch_size $BATCHSIZE --learning_rate $LR --num_epoch $EPOCH --hl && cd ../detector
    python chooseSample.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT --hl
    python find_th.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT --hl
    python detect.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT --hl
else
    cd graphConstruct && ./runStandalone.sh $PROJECT $CVE_ID && cd preprocess && python processGraph.py --project $PROJECT --cve_id $CVE_ID
    cd ../../graphMatrix && python highLightGraph2Matrix.py --project $PROJECT --cve_id $CVE_ID  
    cd ../gmnDetect && python dividedataset.py --project $PROJECT --cve_id $CVE_ID 
    cd gmn && python train_disjoint.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --batch_size $BATCHSIZE --learning_rate $LR --num_epoch $EPOCH
    cd ../detector && python chooseSample.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT 
    python find_th.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT 
    python detect.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT 
fi


