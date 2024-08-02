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
    # 使用空字符串替换所有匹配的注释
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


# if __name__ == "__main__":
#     code = """/*
# Source Path: {src/util.c}
# Source Link: {http://thekelleys.org.uk/gitweb/?p=dnsmasq.git;a=snapshot;h=0549c73b7ea6b22a3c49beb4d432f185a81efcbc}
# Source Commit: {http://thekelleys.org.uk/gitweb/?p=dnsmasq.git;a=commit;h=0549c73b7ea6b22a3c49beb4d432f185a81efcbc}
# */

# unsigned char *do_rfc1035_name(unsigned char *p, char *sval)
# {
#   int j;
#   while (sval && *sval)
#     {
#       unsigned char *cp = p++;
#       for (j = 0; *sval && (*sval != '.'); sval++, j++)
# 	{
# #ifdef HAVE_DNSSEC
# 	  if (option_bool(OPT_DNSSEC_VALID) && *sval == NAME_ESCAPE)
# 	    *p++ = (*(++sval))-1;
# 	  else
# #endif		
# 	    *p++ = *sval;
# 	}
#       *cp  = j;
#       if (*sval)
# 	sval++;
#     }
#   return p;
# }
# /* for use during startup */
# """
    # code_list = [code]
    # "\n".join(code_list)
    # for code in code_list:
    # code = remove_comments(code)
    # variables = get_variables(code)
    # code = unify_var(code, variables)
    # print(code)
