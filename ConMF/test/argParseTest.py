#_-*- coding:utf-8-*-
import argparse
# https://blog.csdn.net/cbbbc/article/details/49904845 这里有详细的对argparse的详细介绍
# parser = argparse.ArgumentParser()
# parser.add_argument("square", help="display a square of a given number",nargs='?',type=int)
# args = parser.parse_args()
# print args.square**2


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
args = parser.parse_args()
if args.verbose:
    print "verbosity turned on"


