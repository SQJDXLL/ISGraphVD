#!/bin/bash

PROJECT=$1
CVE_ID=$2
Change=$3
RL=$4
mission_id=$5


cd "$(dirname "$0")"
cd preprocess
python processPseudo.py --project $PROJECT --cve_id $CVE_ID --RL --mission_id $mission_id

echo "Current working directory: $(pwd)"

cd ../../
echo $(realpath "$0")
script_dir="$(dirname "$(realpath "$0")")"


file_path="$script_dir/standalone-ext/target/universal/stage/bin/standalone"


if [ -e "$file_path" ]; then
    echo "文件已存在，跳过操作。"
else
    echo "仍需要重新stage"
    cd ../
    cd standalone-ext
    sbt clean && sbt stage
    cd ../
fi

echo "Current working directory: $(pwd)"
cd graphConstruct
./standalone-ext/target/universal/stage/bin/standalone $PROJECT $CVE_ID $RL $mission_id


