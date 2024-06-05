# -*- coding: utf-8 -*-
import os,sys,json


from slices import check_SLICES_debug

a="Sm Eu Sm 0 1 Pt 0 1 oo- 0 1 -o- 0 2 -oo 0 2 -oo 0 2 o-o 0 2 ++o 0 0 7 +oo 0 3 ++o 0 3 ++o 1 3 +oo 1 3 o+o 1 3 +oo 1 3 o+o 1 3 ooo 1 2 oo- 2 2 o+o 2 2 +oo 2 2 6 2 2 2 2 2 2 2 2 2 2 3 2"

print(check_SLICES_debug(a,strategy=4,dupli_check=False,graph_rank_check=False))



