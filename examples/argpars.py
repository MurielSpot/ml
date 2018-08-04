# example 1 #########################################################

from argparse import ArgumentParser

'''
ArgumentParser(prog=None, usage=None, description=None, epilog=None)
prog : program 的名字;
usage : 字串，主要是會顯示來告知使用者說應該怎麼使用你寫的;
description : 字串，通常是一段簡短的說明，用來告知使用者說這個程式在做什麼;
epilog : 字串，會出現在參數說明字串的最後面，通常是一些補充資料。
'''
parser=ArgumentParser(prog="here is program name", usage="here is the usage", description="here is the description", epilog="here is the epilog")
parser.print_help()

'''
输出：
usage: here is the usage

here is the description

optional arguments:
  -h, --help  show this help message and exit

here is the epilog
'''

# example 2 #########################################################


# example 3 #########################################################


# example 4 #########################################################



# example 5 #########################################################



# example 6 #########################################################
