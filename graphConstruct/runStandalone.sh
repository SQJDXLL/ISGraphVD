#!/bin/bash

PROJECT=$1
CVE_ID=$2
Change=$3
RL=$4

# path="$(dirname "$0")"
# echo $PATH
cd "$(dirname "$0")"
cd preprocess
python processPseudo.py --project $PROJECT --cve_id $CVE_ID --RL $RL

# cd "$(dirname "$0")"
cd ../
cd standalone-ext
sbt clean && sbt stage
cd ../

# 获取当前脚本的绝对路径
script_dir="$(dirname "$(realpath "$0")")"

echo "当前脚本所在的路径为: $script_dir"
file_path="$script_dir/standalone-ext/target/universal/stage/bin/standalone"

start_time=$(date +%s)

# if [ "$Change" = "True" ]; then
#     sed -i "356a\
#     declare inputPath=$1\n\
#     declare outputPath=$2" ${file_path} 

#     sed -i '261s/$/ \\/' ${file_path}

#     sed -i '261a\
#         "${inputPath}" \\\
#         "${outputPath}"' ${file_path} 
#     # ./standalone-ext/target/universal/stage/bin/standalone
#     ./standalone-ext/target/universal/stage/bin/standalone $RL
# else
#     # 向main.class传参
#     ./standalone-ext/target/universal/stage/bin/standalone $PROJECT $CVE_ID $RL
# fi

if [ "$Change" = "True" ]; then
    sed -i "356a\
    declare inputPath=$1\n\
    declare outputPath=$2\n\
    declare rl=$3" ${file_path}

    sed -i '261s/$/ \\/' ${file_path}

    sed -i '261a\
        "${inputPath}" \\\
        "${outputPath}" \\\
        "${rl}"' ${file_path} 
    ./standalone-ext/target/universal/stage/bin/standalone
else
    # 向main.class传参
    ./standalone-ext/target/universal/stage/bin/standalone $PROJECT $CVE_ID $RL
fi


end_time=$(date +%s)
# 计算运行时间（秒）
execution_time=$((end_time - start_time))

# 打印运行时间
echo "Execution time: $execution_time seconds"


# # cd "$script_dir"
# echo "$script_dir"
# cd preprocess
# echo 111111111111111

# python processGraph.py --project $PROJECT --cve_id $CVE_ID