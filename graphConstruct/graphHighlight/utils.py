import re
from typing import List

def remove_strings(code: str) -> str:
    pattern = r"(\".*?\"|\'.*?\')"
    new_code = re.sub(pattern, "", code, flags=re.DOTALL)
    return new_code


def remove_numbers(code: str) -> str:
    pattern = r"\b\d+\b|\b\d*\.\d+\b"
    new_code = re.sub(pattern, "", code)
    return new_code


def remove_function_declaration(code: str) -> str:
    pattern = r"\w+\s*\*?\s*\w+\s*\(__fastcall\)\s*\w+\s*"
    new_code = re.sub(pattern, "", code)
    return new_code


def remove_key(code: str) -> str:
    keywords = "auto|structbreak|else|switch|case|enum|register|type|def|extern|return|unionconst|continue|for|voiddefault|goto|sizeof|volatiledo|if|static|while|break"
    return re.sub(keywords, "", code)


def get_var(code: str) -> List[str]:
    return re.findall("(?:signed|unsigned)?\s*[_a-zA-Z]+ [*_a-zA-Z0-9]+", code)


def remove_comments(code: str) -> str:
    pattern = r"//.*?$|/\*.*?\*/"
    new_code = re.sub(pattern, "", code, flags=re.MULTILINE | re.DOTALL)
    return new_code


def get_variables(code: str) -> List[str]:
    n = remove_function_declaration(code)
    n = remove_key(n)
    n = remove_numbers(n)
    n = remove_strings(n)
    return [v.split(" ")[-1].replace("*", "") for v in get_var(n)]


def unify_var(code: str, vars: List) -> str:
    return re.sub("|".join(vars), "VARIABLE", code)

