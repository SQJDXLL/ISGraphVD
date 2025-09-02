#!/bin/bash

PROJECT=$1
CVE_ID=$2
Change=$3
RL=$4


cd "$(dirname "$0")"
cd preprocess
python processPseudo.py --project $PROJECT --cve_id $CVE_ID 

cd ../
cd standalone-ext
sbt clean && sbt stage
cd ../

script_dir="$(dirname "$(realpath "$0")")"

echo "当前脚本所在的路径为: $script_dir"
file_path="$script_dir/standalone-ext/target/universal/stage/bin/standalone"

start_time=$(date +%s)

if [ "$Change" = "True" ]; then
    sed -i "356a\
    declare inputPath=$1\n\
    declare outputPath=$2" ${file_path} 

    sed -i '261s/$/ \\/' ${file_path}

    sed -i '261a\
        "${inputPath}" \\\
        "${outputPath}"' ${file_path} 

    ./standalone-ext/target/universal/stage/bin/standalone
else

    ./standalone-ext/target/universal/stage/bin/standalone $PROJECT $CVE_ID $RL 
fi

end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time seconds"