import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入 data.py 文件中的字典
from data.setting_test import vul_dict

# 要添加的元素
key = "abc"
value = {'CVE-2021-22902':0.01}
# key = "dnsmasq"
# value = {'CVE-2021-22903':0.01}

# 添加元素的逻辑
if key not in vul_dict:
    vul_dict[key] = value
else:
    # 如果键已经存在，只更新值
    print("here")
    vul_dict[key].update(value)

with open("../../data/setting_test.py", "w") as f:
    f.write(f"vul_dict = {vul_dict}")

# 重新导入该文件以更新字典
# import importlib
# importlib.reload(setting_test)

# 打印更新后的字典
print(vul_dict)
