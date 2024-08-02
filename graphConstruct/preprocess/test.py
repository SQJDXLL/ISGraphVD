import re

# 原始字符串
original_string = 'label = "DDG: ";\t\r\n=""'

# 使用正则表达式匹配并替换
modified_string = re.sub(r'(DDG: ).*$', r'\1"]', original_string, flags=re.MULTILINE)

print(modified_string)
