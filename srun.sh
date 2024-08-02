#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Usage: $0 <screen_name> <script_arguments...>"
    exit 1
fi

screen_name=$2

# 检查是否存在名为 screen_name 的 screen 会话
if ! screen -list | grep -q "\.${screen_name}"; then
    # 如果不存在，则创建一个新的 screen 会话，并执行命令
    screen -dmS "$screen_name"
fi

# 在指定的 screen 会话中执行命令

screen -S "$screen_name" -X logfile "${screen_name}".log
screen -S "$screen_name" -X log on
screen -S "$screen_name" -X exec bash -c './run.sh "$@"' -- "$@"