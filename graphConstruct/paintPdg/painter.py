# coding = utf-8

import os
import sys
import json
import shutil
import tempfile
from loguru import logger

logger.remove(handler_id=None)
logger.add(
    sink="painter.log",
    format="{time} {level} {message}",
)

'''
    Func: get_graph

    @ Description:
        Get AST, CFG, PDG from c code.
    @ Parameters:
        path: string, path to c source code
        output_path: string, path to the storage folder
    @ Ret:
        N/A
'''


def get_graph(path, output_path):
    f = open(path)
    code = f.read()
    f.close()
    # f_name = os.path.basename(path)

    store_path = output_path
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    c_tmp_path = tempfile.mkstemp(prefix='painter_', suffix='.c')

    cpg_tmp_path = tempfile.mkstemp(prefix='painter_cpg_')

    # ast_tmp_folder = tempfile.mkdtemp(prefix='painter_ast_')
    # cfg_tmp_folder = tempfile.mkdtemp(prefix='painter_cfg_')
    pdg_tmp_folder = tempfile.mkdtemp(prefix='painter_pdg_')


    # os.rmdir(ast_tmp_folder)
    # os.rmdir(cfg_tmp_folder)
    os.rmdir(pdg_tmp_folder)

    c_f = open(c_tmp_path[1], 'wb')
    try:
        c_f.write(code.encode())
    except:
        c_f.write(code)
    c_f.close()


    # base_path = os.path.dirname(os.path.realpath(__file__))
    base_path = "../standalone-ext/joern-inst"
    joernParsePath = os.path.join(base_path, "joern-cli", "joern-parse")
    joernExportPath = os.path.join(base_path, "joern-cli", "joern-export")

    try:
        parse_cmd = 'nice -n -20 {} {} --output {} > /dev/null 2>&1'.format(
           joernParsePath, c_tmp_path[1], cpg_tmp_path[1])
        logger.info('Parsing code. CMD: {}'.format(parse_cmd))
        os.system(parse_cmd)

        # paint_ast_cmd = 'nice -n -20 {} --repr ast --out {} {}  > /dev/null 2>&1'.format(
        #     joernExportPath, ast_tmp_folder, cpg_tmp_path[1]
        # )
        # logger.info('Painting AST Graph. CMD: {}'.format(paint_ast_cmd))
        # os.system(paint_ast_cmd)

        # paint_cfg_cmd = 'nice -n -20 {} --repr cfg --out {} {}  > /dev/null 2>&1'.format(
        #     joernExportPath, cfg_tmp_folder, cpg_tmp_path[1]
        # )
        # logger.info('Painting AST Graph. CMD: {}'.format(paint_cfg_cmd))
        # os.system(paint_cfg_cmd)

        paint_pdg_cmd = 'nice -n -20 {}  --repr pdg --out {} {}  > /dev/null 2>&1'.format(
            joernExportPath, pdg_tmp_folder, cpg_tmp_path[1]
        )
        logger.info('Painting PDG Graph. CMD: {}'.format(paint_pdg_cmd))
        os.system(paint_pdg_cmd)


        # shutil.move('{}/0-ast.dot'.format(ast_tmp_folder), '{}/ast.dot'.format(store_path))
        # shutil.move('{}/1-cfg.dot'.format(cfg_tmp_folder), '{}/cfg.dot'.format(store_path))
        shutil.move('{}/1-pdg.dot'.format(pdg_tmp_folder), '{}/pdg.dot'.format(store_path))

        

    except Exception as e:
        logger.error('Failed to generate graph. Reason: {}'.format(str(e)))

    os.remove(c_tmp_path[1])
    os.remove(cpg_tmp_path[1])

    # shutil.rmtree(ast_tmp_folder)
    # shutil.rmtree(cfg_tmp_folder)
    shutil.rmtree(pdg_tmp_folder)

    
if __name__ == '__main__':
    get_graph(sys.argv[1], 'output')