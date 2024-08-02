# #!/bin/bash

# # 一键启动s
# # chmod +x run.sh


# # ./run.sh busybox CVE-2018-20679 1 32 1e-4 50 True True False > CVE-2018-20679.log &
# # ./run.sh curl CVE-2019-5482 2 32 1e-4 10 True True > CVE-2019-5482.log &
# # ./run.sh curl CVE-2021-22901 0 32 1e-4 10 True True > CVE-2021-22901.log &

# # ./run.sh curl CVE-2019-5482 2 32 1e-4 50 True True False > CVE-2019-5482.log &
# ./run.sh miniupnpc CVE-2015-6031 0 32 1e-4 20 True True False > CVE-2015-6031.log &


# 近物所
# ./run.sh dnsmasq CVE-2015-8899 0 32 5e-5 100 True True True > CVE-2015-8899.log &
# ./run.sh dnsmasq CVE-2017-14491 1 32 1e-4 20 True True False > CVE-2017-14491.log &
# ./run.sh hostapd CVE-2019-16275 2 32 1e-4 50 True True False > CVE-2019-16275.log &
# ./run.sh miniupnpd CVE-2017-1000494 1 32 1e-4 20 True True False > CVE-2017-1000494.log &


# ./srun.sh dnsmasq CVE-2015-8899 2 32 5e-5 100 True True False 
# ./srun.sh hostapd CVE-2019-16275 1 32 1e-4 50 True False False 
# ./srun.sh dnsmasq CVE-2017-14491 5 32 1e-4 20 True False False 
# ./srun.sh miniupnpd CVE-2017-1000494 2 32 1e-4 20 True False False
# ./srun.sh miniupnpc CVE-2015-6031 6 32 1e-4 50 True False True
# ./srun.sh curl CVE-2021-22901 6 32 1e-4 20 True False False

# ./srun.sh busybox CVE-2018-20679 6 32 1e-4 20 True False False
# ./srun.sh curl CVE-2019-5482 6 32 1e-4 20 True True False


# 原来的
# ./srun.sh miniupnpc CVE-2015-6031 3 32 1e-4 10 True False False

# new 近物所 0311
# ./srun.sh busybox CVE-2018-20679 1 32 1e-4 20 True True False
# ./srun.sh curl CVE-2019-5482 2 32 5e-5 50 True True False

# new 98 0311
# ./srun.sh busybox CVE-2018-20679 0 32 1e-4 20 True True False
# ./srun.sh curl CVE-2019-5482 1 32 1e-4 20 True True False

# 近物所
# ./srun.sh busybox CVE-2018-20679 1 32 1e-4 50 True False False
# ./srun.sh curl CVE-2019-5482 2 32 1e-4 50 True False False


# # tail -f CVE-2015-8899.log

# # 查看程序是否还在运行，并且可以看到PID,可以使用kill PID停止进程
# # ps aux | grep ./run.sh





PROJECT=$1
CVE_ID=$2
GPU=$3
BATCHSIZE=$4
LR=$5
EPOCH=$6
LARGEORNOT=$7
HL=$8
Change=$9 # change代表是否修改standalone的代码，是否需要重新编译
RL=${10}


eval "$(conda shell.bash hook)"
conda activate zyl_new


# First
# 需要修改graphMatrix 下面的config文件和运行代码中的一个地方(divide_by_datatype函数)



# 8798涉及到变量的修改只是一个条件判断的增加，因此，和8899一样也需要5e-5
# ./srun.sh miniupnpc CVE-2017-8798 1 16 5e-5 50 True True False

# ./srun.sh dnsmasq CVE-2021-3448 2 32 5e-5 50 True False False
# ./srun.sh miniupnpd CVE-2019-12110 3 32 1e-4 20 True True False

# ./srun.sh curl CVE-2023-46218 4 4 1e-4 50 True False False 
# ./srun.sh dnsmasq CVE-2017-13704 4 32 1e-4 50 True False False


# ./srun.sh dnsmasq CVE-2019-14834 1 8 1e-4 50 True False False
# ./srun.sh curl CVE-2023-46219 0 32 1e-4 10 True True False
# ./srun.sh dnsmasq CVE-2017-14496 2 32 1e-4 50 True True False



# 0407
# long
# ./srun.sh curl CVE-2023-46218 0 32 5e-5 100 True True False 

# ./srun.sh curl CVE-2019-14834 1 16 1e-4 50 True True False
# mipsel_clang-12_O0_curl_fix_Curl_cookie_add.c mipsel_clang-12_O0_curl_vul_Curl_cookie_add.c
#  mips_clang-12_O0_curl_fix_Curl_cookie_add.c mips_clang-12_O0_curl_vul_Curl_cookie_add.c


# ./srun.sh dnsmasq CVE-2017-13704 4 32 1e-4 50 True True False

# normal
# ./srun.sh miniupnpc CVE-2017-8798 2 16 5e-5 50 True True False

# ./srun.sh dnsmasq CVE-2021-3448 3 32 5e-5 50 True True False
# ./srun.sh miniupnpd CVE-2019-12110 5 32 1e-4 50 True True False
# ./srun.sh curl CVE-2023-46219 6 32 1e-4 10 True True False
# ./srun.sh dnsmasq CVE-2017-14496 6 32 1e-4 20 True True False

# 重训练
# ./srun.sh miniupnpd CVE-2019-12110 5 32 5e-5 50 True True False
# 先基础实验验证一下
# ./srun.sh miniupnpc CVE-2017-8798 2 32 5e-5 50 True False False
# ./srun.sh curl CVE-2023-46218 0 32 5e-5 100 True False False 
# ./srun.sh curl CVE-2019-14834 1 32 5e-5 100 True False False


# if [ "$HL" = "True" ]; then
#     # 添加指令
#     cd graphConstruct && ./runStandalone.sh $PROJECT $CVE_ID $Change && cd preprocess && python processGraph.py --project $PROJECT --cve_id $CVE_ID && \
#     cd ../graphHighlight && python makeWinandSlice.py --project $PROJECT --cve_id $CVE_ID && \
#     python diff2Dot.py --project $PROJECT --cve_id $CVE_ID && \
#     python drawHight.py --project $PROJECT --cve_id $CVE_ID && cd ../../graphMatrix && \
#     python highLightGraph2Matrix.py --project $PROJECT --cve_id $CVE_ID --hl && cd ../gmnDetect && \
#     python dividedataset.py --project $PROJECT --cve_id $CVE_ID --hl && cd gmn && \
#     python train_disjoint.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --batch_size $BATCHSIZE --learning_rate $LR --num_epoch $EPOCH --hl && \
#     cd ../detector && python chooseSample.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT --hl && \
#     python find_th.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT --hl && \
#     python detect.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT --hl
# else
#     cd graphConstruct && ./runStandalone.sh $PROJECT $CVE_ID && cd preprocess && python processGraph.py --project $PROJECT --cve_id $CVE_ID && \
#     cd ../../graphMatrix && python highLightGraph2Matrix.py --project $PROJECT --cve_id $CVE_ID && \
#     cd ../gmnDetect && python dividedataset.py --project $PROJECT --cve_id $CVE_ID && \
#     cd gmn && python train_disjoint.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --batch_size $BATCHSIZE --learning_rate $LR --num_epoch $EPOCH && \
#     cd ../detector && python chooseSample.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT && \
#     python find_th.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT && \
#     python detect.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT
# fi

# delete AST
# ./srun.sh busybox CVE-2018-20679 1 32 1e-4 50 True False False
# ./srun.sh curl CVE-2019-5482 2 32 1e-4 50 True False False
# ./srun.sh hostapd CVE-2019-16275 1 32 1e-4 50 True False False
# ./srun.sh dnsmasq CVE-2017-14491 2 32 1e-4 50 True False False


if [ "$HL" = "True" ]; then
    if [ "$RL" = "True" ]; then
        cd graphConstruct/ && ./runStandalone.sh $PROJECT $CVE_ID $Change $RL && \
        cd paintPdg && python paintBatch.py --project $PROJECT --cve_id $CVE_ID --RL $RL && \
        cd ../preprocess && python processGraph.py --project $PROJECT --cve_id $CVE_ID --RL $RL && \
        cd ../graphHighlight && python makeWinandSlice.py --project $PROJECT --cve_id $CVE_ID --RL  && \
        python diff2Dot.py --project $PROJECT --cve_id $CVE_ID --RL && \
        python drawHight.py --project $PROJECT --cve_id $CVE_ID --RL && cd ../../graphMatrix  && \
        python highLightGraph2Matrix.py --project $PROJECT --cve_id $CVE_ID --hl --RL && cd ../gmnDetect && \
        python dividedataset.py --project $PROJECT --cve_id $CVE_ID --hl --RL && cd detector && \
        python detect_rl.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT --hl
    else
        cd graphConstruct/preprocess/ && python removeDupilcate.py --project $PROJECT --cve_id $CVE_ID && \
        cd ../ && ./runStandalone.sh $PROJECT $CVE_ID $Change && \ 
        cd paintPdg && python paintBatch.py --project $PROJECT --cve_id $CVE_ID && \
        cd ../preprocess && python processGraph.py --project $PROJECT --cve_id $CVE_ID && \
        cd ../graphHighlight && python makeWinandSlice.py --project $PROJECT --cve_id $CVE_ID && \
        python diff2Dot.py --project $PROJECT --cve_id $CVE_ID && \
        python drawHight.py --project $PROJECT --cve_id $CVE_ID && cd ../../graphMatrix && \
        python highLightGraph2Matrix.py --project $PROJECT --cve_id $CVE_ID --hl && cd ../gmnDetect && \
        python dividedataset.py --project $PROJECT --cve_id $CVE_ID --hl && cd gmn && \
        python train_disjoint.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --batch_size $BATCHSIZE --learning_rate $LR --num_epoch $EPOCH --hl && \
        cd ../detector && python chooseSample.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT --hl && \
        python find_th.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT --hl && \
        python detect.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT --hl
    fi
else
    cd graphConstruct/preprocess/ && python removeDupilcate.py --project $PROJECT --cve_id $CVE_ID && \
    cd ../ && ./runStandalone.sh $PROJECT $CVE_ID $Change && \ 
    cd paintPdg && python paintBatch.py --project $PROJECT --cve_id $CVE_ID && \
    cd ../preprocess && python processGraph.py --project $PROJECT --cve_id $CVE_ID && \
    # python deleteAst.py --project $PROJECT --cve_id $CVE_ID && \
    cd ../../graphMatrix && python highLightGraph2Matrix.py --project $PROJECT --cve_id $CVE_ID && \
    cd ../gmnDetect && python dividedataset.py --project $PROJECT --cve_id $CVE_ID && \
    cd gmn && python train_disjoint.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --batch_size $BATCHSIZE --learning_rate $LR --num_epoch $EPOCH && \
    cd ../detector && python chooseSample.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT && \
    python find_th.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT && \
    python detect.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT
fi

# cd graphConstruct/preprocess/ && python removeDupilcate.py --project $PROJECT --cve_id $CVE_ID && \
# cd ../ && ./runStandalone.sh $PROJECT $CVE_ID $Change && \ 
# cd paintPdg && python paintBatch.py --project $PROJECT --cve_id $CVE_ID && \
# cd ../preprocess && python processGraph.py --project $PROJECT --cve_id $CVE_ID && \
# # python deleteAst.py --project $PROJECT --cve_id $CVE_ID && \
# cd ../../graphMatrix && python highLightGraph2Matrix.py --project $PROJECT --cve_id $CVE_ID && \
# cd ../gmnDetect && python dividedataset.py --project $PROJECT --cve_id $CVE_ID && \
# cd gmn && python train_disjoint.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --batch_size $BATCHSIZE --learning_rate $LR --num_epoch $EPOCH && \
# cd ../detector && python chooseSample.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT && \
# python find_th.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT && \
# python detect.py --project $PROJECT --cve_id $CVE_ID --gpu $GPU --largeOrnot $LARGEORNOT


# pg-vulnet
# ./srun.sh hostapd CVE-2019-16275 1 64 1e-4 20 True False False
# ./srun.sh miniupnpd CVE-2017-1000494 2 32 1e-4 20 True False False
# ./srun.sh busybox CVE-2018-20679 1 64 1e-4 20 True False False
# ./srun.sh curl CVE-2021-22901 2 64 1e-4 20 True False False 
# ./srun.sh dnsmasq CVE-2017-14491 2 64 1e-4 20 True False False
# ./srun.sh dnsmasq CVE-2015-8899 1 32 5e-5 100 True False False
# ./srun.sh miniupnpc CVE-2015-6031 2 64 1e-4 20 True False False

# ./srun.sh curl CVE-2019-5482 1 32 1e-4 20 True False False


# finish
# ./srun.sh miniupnpd CVE-2017-1000494 1 32 1e-4 20 True True False
# ./srun.sh miniupnpc CVE-2015-6031 2 64 1e-4 20 True True False  
# ./srun.sh busybox CVE-2018-20679 1 64 1e-4 20 True True False
# ./srun.sh curl CVE-2021-22901 2 64 1e-4 20 True True False 
# ./srun.sh dnsmasq CVE-2017-14491 2 64 1e-4 20 True True False 
# ./srun.sh curl CVE-2019-5482 1 64 1e-4 20 True True False
# ./srun.sh dnsmasq CVE-2017-14496 2 64 1e-4 20 True True False
# ./srun.sh curl CVE-2023-46219 2 64 1e-4 20 True True False
# ./srun.sh miniupnpd CVE-2019-12110 2 32 1e-4 20 True True False
# ./srun.sh hostapd CVE-2019-16275 1 64 1e-4 20 True True False


# run 

# ./srun.sh dnsmasq CVE-2021-3448 2 32 5e-5 100 True True False


# ./srun.sh dnsmasq CVE-2017-13704 1 32 1e-4 50 True True False
# ./srun.sh dnsmasq CVE-2015-8899 1 32 5e-5 100 True True False



# ckpt
# python3 chooseSample_ckpt.py  --project $PROJECT --cve_id $CVE_ID --gpu $GPU --batch_size $BATCHSIZE --learning_rate $LR --num_epoch $EPOCH --largeOrnot $LARGEORNOT --hl
#     python3 find_th_ckpt.py  --project $PROJECT --cve_id $CVE_ID --gpu $GPU --batch_size $BATCHSIZE --learning_rate $LR --num_epoch $EPOCH --largeOrnot $LARGEORNOT --hl
    # python3 detect_ckpt.py  --project $PROJECT --cve_id $CVE_ID --gpu $GPU --batch_size $BATCHSIZE --learning_rate $LR --num_epoch $EPOCH --largeOrnot $LARGEORNOT --hl