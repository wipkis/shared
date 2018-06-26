#
# This example demonstrates usage of the included Python grammars
#

import sys
import os, os.path
from io import open
import glob, time

from lark import Lark
from lark.indenter import Indenter
from lark.lexer import Token
from lark.tree import Tree

__path__ = os.path.dirname(__file__)


class PythonIndenter(Indenter):
    NL_type = '_NEWLINE'
    OPEN_PAREN_types = ['__LPAR', '__LSQB', '__LBRACE']
    CLOSE_PAREN_types = ['__RPAR', '__RSQB', '__RBRACE']
    INDENT_type = '_INDENT'
    DEDENT_type = '_DEDENT'
    tab_len = 8


grammar3_filename = os.path.join(__path__, 'python3.g')
with open(grammar3_filename) as f:
    python_parser3 = Lark(f, parser='lalr', postlex=PythonIndenter(), start='file_input')


def _read(fn, *args):
    kwargs = {'encoding': 'iso-8859-1'}
    with open(fn, *args, **kwargs) as f:
        return f.read()


class PythonParser:
    vallist = ['val1', 'val2', 'val3', 'val4', 'val5', 'val6', 'val7', 'val8', 'val9', 'val10', 'val11', 'val12',
               'val13', 'val14', 'val15', 'val16', 'val17', 'val18', 'val19', 'val20', 'val21', 'val22', 'val23',
               'val24', 'val25', 'val26', 'val27', 'val28', 'val29', 'val30', 'val31', 'val32', 'val33', 'val34',
               'val35', 'val36', 'val37', 'val38', 'val39', 'val40', 'val41', 'val42', 'val43', 'val44', 'val45',
               'val46', 'val47', 'val48', 'val49', 'val50', 'val51', 'val52', 'val53', 'val54', 'val55', 'val56',
               'val57', 'val58', 'val59', 'val60']

    def __init__(self):
        print('무언가')
        self.namedict = {'val1': '', 'val2': '', 'val3': '', 'val4': '', 'val5': '', 'val6': '', 'val7': '', 'val8': '',
                         'val9': '', 'val10': '', 'val11': '', 'val12': '', 'val13': '', 'val14': '', 'val15': '',
                         'val16': '', 'val17': '', 'val18': '', 'val19': '', 'val20': '', 'val21': '', 'val22': '',
                         'val23': '', 'val24': '', 'val25': '', 'val26': '', 'val27': '', 'val28': '', 'val29': '',
                         'val30': '', 'val31': '', 'val32': '', 'val33': '', 'val34': '', 'val35': '', 'val36': '',
                         'val37': '', 'val38': '', 'val39': '', 'val40': '', 'val41': '', 'val42': '', 'val43': '',
                         'val44': '', 'val45': '', 'val46': '', 'val47': '', 'val48': '', 'val49': '', 'val50': '',
                         'val51': '', 'val52': '', 'val53': '', 'val54': '', 'val55': '', 'val56': '', 'val57': '',
                         'val58': '', 'val59': '', 'val60': ''}
        self.inversedic = {}
        self.wordnum = 0
        self.debug = False

    def add_dict(self, value):
        key = PythonParser.vallist[self.wordnum]
        if self.inversedic.get(value, None) is None:
            if self.debug is True:
                try:
                    print(value, '->', key)
                except UnicodeEncodeError:
                    print('value is not cp949')
            self.namedict[key] = value
            self.inversedic[value] = key
            self.wordnum += 1
            if self.wordnum is 60:
                self.wordnum = 59
            return key
        return self.inversedic.get(value, None)

    def clear_dict(self):
        self.wordnum = 0
        self.namedict = {'val1': '', 'val2': '', 'val3': '', 'val4': '', 'val5': '', 'val6': '', 'val7': '', 'val8': '',
                         'val9': '', 'val10': '', 'val11': '', 'val12': '', 'val13': '', 'val14': '', 'val15': '',
                         'val16': '', 'val17': '', 'val18': '', 'val19': '', 'val20': '', 'val21': '', 'val22': '',
                         'val23': '', 'val24': '', 'val25': '', 'val26': '', 'val27': '', 'val28': '', 'val29': '',
                         'val30': '', 'val31': '', 'val32': '', 'val33': '', 'val34': '', 'val35': '', 'val36': '',
                         'val37': '', 'val38': '', 'val39': '', 'val40': '', 'val41': '', 'val42': '', 'val43': '',
                         'val44': '', 'val45': '', 'val46': '', 'val47': '', 'val48': '', 'val49': '', 'val50': '',
                         'val51': '', 'val52': '', 'val53': '', 'val54': '', 'val55': '', 'val56': '', 'val57': '',
                         'val58': '', 'val59': '', 'val60': ''}
        self.inversedic.clear()

    def python_to_tree(self, file):
        ast = None
        try:
            ast = python_parser3.parse(_read(file) + '\n')
        except IndexError:
            print('object file does not given')
        return ast

    def tree_to_list(self, ast):
        # 나온 ast를 전위 탐색
        q = [ast]
        ret = []
        while len(q) > 0:
            t = q.pop()
            if isinstance(t, Tree):
                ret.append(t.data)
                # stack이므로 역순으로 넣는다
                if isinstance(t.children, list):
                    q.append('treeout_stmt')
                    num = len(t.children)
                    for i in range(num):
                        q.append(t.children[num - i - 1])
                    q.append('treein_stmt')
                else:
                    q.append(t.children)
            elif isinstance(t, Token):
                # 토큰이면 type이나 value중 필요한 것만 넣는다.
                # number 계열이면 그걸 빼야 하는데...
                token = t.type == 'NAME' or t.type == 'DEC_NUMBER' or t.type == 'FLOAT_NUMBER' \
                        or t.type == 'HEX_NUMBER' or t.type == 'OCT_NUMBER' or t.type == 'IMAG_NUMBER' \
                        or t.type == 'BIN_NUMBER' or t.type == 'STRING' or t.type == 'LONG_STRING'
                if token is True:
                    ret.append(self.add_dict(t.value))
                else:
                    ret.append(t.type)
            elif isinstance(t, str):
                ret.append(t)
            else:
                print('something', type(t))
        return ret


def ret_parser():
    return PythonParser()


if __name__ == '__main__':
    #argv1 받아서 해당 파일을 파싱하고 pretty로 출력
    ast = None
    parser = PythonParser()
    try:
        ast = parser.python_to_tree(sys.argv[1])
        print(ast.pretty())
    except IndexError:
        print('object file does not given')
    except UnicodeEncodeError:
        print('maybe a not str?')
    ret = parser.tree_to_list(ast)
    print(ret)
    exit()
