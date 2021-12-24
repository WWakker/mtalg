# import os
# os.environ['CDSW_NODE_NAME'] = 'CDSW_NODE_NAME'
# os.environ['DISC_KRB_REALM'] = 'MIT01.ECB.DE'
# os.environ['CDSW_NODE_NAME'] = 'CDSW_NODE_NAME'
# os.environ['CDSW_CPU_MILLICORES'] = '1000'
# print(os.name)


# from mtalg.random import MultithreadedRNG
# import numpy as np


# class TestMRNGCDSW:
#     def test1(self):
#         mrng = MultithreadedRNG(seed=1)
#         assert mrng.values.size == 0
#         assert mrng.num_threads == 1

#     def test2(self):
#         mrng = MultithreadedRNG(seed=1, num_threads=4)
#         mrng.standard_normal(4)
#         assert mrng.values.shape == (4,)
#         assert mrng.num_threads == 1
