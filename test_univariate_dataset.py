from test_MILOF import test

import multiprocessing as mp
from multiprocessing import Manager

pool =mp.Pool(mp.cpu_count())
output =pool.starmap(test, ["MILOF","iforestASD", "HStree-tree","ARIMAFD","KitNet"] )


