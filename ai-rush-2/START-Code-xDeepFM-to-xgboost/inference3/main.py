#from data_local_loader import get_data_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import argparse
import numpy as np
import time
import datetime
import pandas as pd
import pickle
# xDeepFM model
import tensorflow as tf
from deepctr.xdeepfm import xDeepFM
from deepctr.inputs import SparseFeat,DenseFeat,get_fixlen_feature_names
from sklearn.metrics import mean_squared_error,accuracy_score, auc, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_loader import feed_infer
from evaluation import evaluation_metrics
import nsml
import keras
import math
    

from multiprocessing import Pool
import time
from tqdm import tqdm

from Resnet_feature_extractor import resnet_feature_extractor

if not nsml.IS_ON_NSML:
    DATASET_PATH = os.path.join('/airush2_temp')
    DATASET_NAME = 'airush2_temp'
    print('use local gpu...!')
    use_nsml = False
else:
    from nsml import DATASET_PATH, DATASET_NAME, NSML_NFS_OUTPUT, SESSION_NAME
    
    DATASET_PATH = os.path.join(nsml.DATASET_PATH)
    print('start using nsml...!')
    print('DATASET_PATH: ', DATASET_PATH)
    use_nsml = True


fixlen_feature_names_global=[]
lit_cnt_prob_list = '0.0,0.0833,0.0143,0.0,0.0909,0.0,0.0426,0.1667,0.0644,0.0,0.0,0.0,0.029,0.0,0.0,0.0,0.0,0.0414,0.0,0.0,0.0403,0.0238,0.0,0.0,0.5,0.1818,0.0226,0.2174,0.0,0.015,0.0569,0.0,0.0,0.0,0.0747,0.0,0.0,0.0714,0.0229,0.0,0.0803,0.1582,0.0,0.045,0.037,0.0278,0.0887,0.0,0.0,0.0,0.0725,0.0473,0.0,0.1501,0.0417,0.0,0.0,0.0,0.1538,0.0606,0.0754,0.0,0.0155,0.0159,0.0,0.0,0.0,0.0,0.0638,0.0667,0.0,0.0721,0.0,0.0776,0.0,0.0,0.007,0.0829,0.0,0.1911,0.0,0.1515,0.1339,0.0504,0.1055,0.0914,0.0,0.0,0.0,0.0333,0.0,0.0789,0.0303,0.1824,1.0,0.0,0.0434,0.1207,0.0,0.1189,0.0128,0.069,0.0,0.0,0.013,0.2,0.0976,0.0573,0.0,0.0,0.0227,0.0,0.0299,0.0655,0.0,0.0779,0.101,0.0,0.0,0.0,0.0645,0.058,0.1128,0.0,0.0,0.0,0.0,0.0,0.045,0.0,0.0,0.0882,0.1148,0.0,0.0,0.0204,0.0769,0.0,0.1,0.1056,0.0187,0.0,0.0315,0.0,0.0471,0.1351,0.0,0.0,0.0,0.048,0.0345,0.0,0.0,0.0531,0.0,0.0771,0.0,0.0,0.0,0.0,0.1429,0.0046,0.0602,0.0,0.0482,0.5,0.0,0.0,0.0297,0.0,0.0,0.083,0.027,0.0,0.1435,0.1034,0.0,0.0182,0.0,0.0,0.0,0.05,0.0588,0.0721,0.0741,0.0928,0.0,0.0952,0.147,0.1333,0.0,0.125,0.0,0.0909,0.1274,0.0375,0.024,0.0345,1.0,0.0,0.0,0.0,0.1266,0.1019,0.0769,0.0,0.0484,0.0,0.1842,0.0588,0.0433,0.0282,0.0526,0.0889,0.1156,0.0,0.027,0.1429,0.0984,0.086,0.0,0.0316,0.1168,0.106,0.0475,0.0558,0.0,0.1683,0.0,0.0,0.0,0.0,0.0,0.0435,0.0,0.0,0.0,0.0,0.1338,0.0271,0.0,0.0846,0.0726,0.0,0.0,0.0586,0.0,0.0649,0.0227,0.0292,0.0159,0.0,0.1009,0.3333,0.0,0.0952,0.0,0.0909,0.0714,0.0,0.0,0.0,0.1198,0.0,0.0614,0.0556,0.0,0.0115,0.2222,0.1382,0.0363,0.0,0.0,0.0,0.0,0.0,0.0909,0.0442,0.1145,0.0149,0.0,0.0417,0.0,0.0135,0.0889,0.0,0.0179,0.1481,0.0,0.0909,0.0,0.0,0.0438,0.1429,0.1005,0.013,0.0,0.0,0.0,0.0,0.0,0.1364,0.0,0.0334,0.1085,0.0,0.0,0.0,0.0488,0.0705,0.0,0.0,0.0621,1.0,0.0,0.0,0.0491,0.0,0.0771,0.063,0.0473,0.0,0.0,0.0,0.0,0.0668,0.0394,0.0786,0.0526,0.5,0.011,0.125,0.0,0.0267,0.0,0.0656,0.121,0.0,0.0,0.027,0.0,0.0455,0.1273,0.0571,0.0,0.0,0.0933,0.0,0.0435,0.0,0.1429,0.0,0.0,0.0845,0.0169,0.1943,0.0,0.0345,0.25,0.0,0.0,0.0,0.0,0.0,0.0811,0.0,0.1013,0.0,0.0,0.0,0.0,0.1601,0.0,0.048,0.1429,0.0,0.0238,0.0,0.0366,0.0,0.0,0.0,0.0,0.0,0.0588,0.0833,0.0,0.25,0.0951,0.0,0.1059,0.0,0.0,0.0769,0.0589,0.0,0.0,0.0467,0.0,0.0,0.0357,0.0,0.0276,0.1,0.0,0.1429,0.0,0.0,0.0,0.0,0.0235,0.0,0.038,0.0233,0.0513,0.0395,0.027,0.3333,0.2,0.0,0.0,0.0566,0.0,0.0,0.0239,0.0167,0.0,0.0,0.0,0.0,0.0,0.0,0.043,0.0,0.0,0.0,0.0482,0.0575,0.0,0.0,0.0,0.0588,0.0975,0.0,0.0,0.0,0.1253,0.0,0.1351,0.0461,0.0679,0.0,0.087,0.0966,0.0653,0.0,0.0,0.0,0.0,0.0,0.0,0.0466,0.0,0.0,0.1008,0.0264,0.0,0.1147,0.0484,0.1212,0.0,1.0,0.0559,0.0287,0.0284,0.0354,0.0211,0.04,0.0526,0.0526,0.073,0.103,0.75,0.0,0.0,0.0694,0.0,0.0526,0.0,0.0,0.0,0.0507,0.0934,0.1176,0.0,0.0,0.0,0.0,0.0481,0.1414,0.0,0.0149,0.5,0.0,0.0,0.0,0.0,0.0375,0.0203,0.0,0.0405,0.0,0.0,0.0,0.0171,0.0,0.0,0.0476,0.0212,0.0714,0.0,0.0,0.1111,0.0,0.0625,0.0,0.0654,0.0,0.0798,0.0,0.0,0.0378,0.0,0.0,0.0476,0.1514,0.0,0.0,0.1667,0.0103,0.0,0.0,0.0,0.0,0.0363,0.0323,0.0606,0.0667,0.0,0.0,0.0751,0.0,0.0,0.0,0.0618,0.0,0.0,0.1277,0.0,0.25,0.0472,0.1304,0.0,0.0303,0.095,0.0,0.0211,0.1667,0.0625,0.2222,0.0,0.0,0.0,0.2857,0.0,0.0664,0.0,0.0589,0.1213,0.0556,0.0909,0.0,0.0365,0.0,0.0,0.0241,0.0,0.0,0.0,0.0,0.0276,0.0,0.1527,0.0,0.0,0.0,0.0256,0.0,0.0667,0.0174,0.0,0.0,0.0,0.058,0.0145,0.0,0.1067,0.0123,0.0312,0.0,0.0,0.135,0.0,0.0,0.0,0.0,0.0,0.0,0.0769,0.0,0.1429,0.0158,0.0,0.0,0.0,0.0,0.0,0.0,0.0928,0.0652,0.1429,0.1152,0.0,0.0226,0.0,0.0778,0.0652,0.0,0.0,0.0,0.0392,0.069,0.0,0.0,0.1538,0.0,0.0751,0.0,0.0625,0.0,0.0,0.0964,0.0,0.0,0.1568,0.0658,0.0087,0.0492,0.0,0.0,0.0,0.0236,0.0,0.0,0.0334,0.0643,0.0,0.2044,0.0,0.089,0.0,0.0,0.0,0.0,0.0,0.0,0.1134,0.0,0.0,0.0757,0.0486,0.0499,0.0,0.1,0.0,0.078,0.0,0.0851,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1383,0.0,0.054,0.0,0.0,0.0,0.0061,0.0,0.0605,0.1111,0.0,0.0446,0.0212,0.0,0.0,0.0503,0.0,0.0,0.0,0.0,0.0,0.0,0.1231,0.0,0.0,0.0,0.0,0.0,0.0227,0.2857,0.0769,0.0,0.1111,0.0809,0.0851,0.0,0.0,0.0,0.0263,0.0,0.0,0.0176,0.0,0.0,0.02,0.0,0.1058,0.094,0.0,0.0,0.0658,0.0971,0.0,0.0384,0.0588,0.0,0.0,0.0306,0.0,0.0628,0.08,0.2857,0.0821,0.134,0.1739,0.0392,0.0,0.0537,0.2,0.0408,0.0962,0.1667,0.0,0.0,0.0526,0.1111,0.1313,0.1905,0.2315,0.0,0.0,0.0441,0.0,0.0,0.25,0.1389,0.0,0.0405,0.0,0.0,0.0,0.0,0.0453,0.0,0.0099,0.0,0.0,0.0,0.0513,0.1278,0.2143,0.0,0.0,0.0269,0.1629,0.0845,0.0675,0.0,0.0619,0.0152,0.0977,0.0,0.1572,0.1,0.0929,0.0,0.0,0.0,0.0085,0.0525,0.0,0.0302,0.1331,0.0,0.0,0.0,0.0723,0.0,0.0787,0.0,0.0,0.0,0.0,0.0855,0.039,0.0142,0.1334,0.0285,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.25,0.0433,0.0,0.1538,0.0,0.3333,0.0789,0.021,0.0,0.0394,0.0588,0.0,0.0914,0.0,0.0,0.144,0.0,0.1184,0.0,0.0,0.0,0.0936,0.0,0.0,0.0,0.0,0.0,0.0769,0.004,0.0,0.02,0.0,0.0,0.1429,0.0333,0.0,0.0,0.0724,0.0,0.0,0.0504,0.0222,0.1123,0.0,0.1029,0.0625,0.0,0.0,0.0,0.0088,0.1564,0.0,0.0,0.1678,0.0,0.0,0.0,0.0,0.0,0.0,0.1298,0.0,0.0303,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0106,0.0,0.1304,0.0,0.0,0.0,0.0,0.017,0.0784,0.0,0.0,0.0,0.0,0.0377,0.1048,0.1429,0.1052,0.0278,0.2222,0.0807,0.0679,0.0,0.0256,0.0,0.0,0.0955,0.043,0.0,0.0833,0.0,0.0411,0.0,0.0,0.0,0.234,0.0313,0.0,0.0,0.0,0.0,0.0486,0.0263,0.0772,0.0,0.0,0.0799,0.0,0.1314,0.3333,0.0,0.05,0.0943,0.0378,0.0,0.0,0.1429,0.0,0.0645,0.0,0.1429,0.0211,0.05,0.5,0.0799,0.0,0.0,0.0,0.0175,0.0,0.5,0.0377,0.1187,0.0,0.0,0.1404,0.037,0.0,0.0158,0.0,0.0,0.1,0.0351,0.0,0.0754,0.0357,0.0,0.0962,0.0,0.0,0.0893,0.0,0.0,0.2162,0.0567,0.0465,0.0,0.0,0.0,0.0,0.1429,0.0,0.0449,0.0345,0.0,0.0,0.0435,0.0509,0.0525,0.5,0.0684,0.0,0.034,0.1458,0.1176,0.0383,1.0,0.3043,0.0,0.0417,0.0312,0.0645,0.0,0.0,0.0461,0.0,0.0,0.0,0.0,0.104,0.0,0.0891,0.1063,0.0,0.0565,0.131,0.0,0.0,0.0055,0.0105,0.0,0.037,0.0723,0.0,0.0,0.1002,0.027,0.0276,0.0,0.0274,0.0,0.0414,0.0,0.0,0.0936,0.0,0.0196,0.0,0.0455,0.0,0.0,0.1098,0.0,0.0,0.0865,0.0938,0.0539,0.0727,0.0,0.0,0.0,0.0404,0.0961,0.0667,0.0981,0.0,0.0,0.037,0.0368,0.0821,0.2,0.0,0.0701,0.0587,0.0283,0.0103,0.0294,0.0927,0.1056,0.0146,0.0,0.0,0.0317,0.0,0.0,0.0,0.0,0.0421,0.1429,0.0,0.03,0.0,0.0283,0.0,0.0633,0.0,0.098,0.0588,0.0,0.0,0.0435,0.0614,0.0282,0.0,0.0,0.0,0.1419,0.0,0.0,0.0227,0.1167,0.0963,0.0725,0.0536,0.036,0.0,0.0851,0.0635,0.0208,0.0,0.0435,0.0754,0.0,0.1019,0.0,0.095,0.0,0.0593,0.0856,0.029,0.1319,0.0,0.0927,0.0043,0.0,0.1133,0.0,0.0,0.0,0.0,0.0625,0.06,0.0,0.0408,0.087,0.1667,0.0,0.12,0.0153,0.0,0.0232,0.0339,0.0192,0.0,0.0648,0.0,0.0513,0.5,0.3333,0.0625,0.0621,0.0,0.0,0.2,0.1153,0.1844,0.0,0.0,0.0,0.0,0.0342,0.0,0.0829,0.0,0.0319,0.0,0.0,0.0,0.0,0.0832,0.0822,0.0,0.1034,0.0,0.1667,0.0282,0.0777,0.25,0.0,0.0,0.0566,0.0718,0.0529,0.0,0.0,0.0,0.0,0.0291,0.0,0.0,0.0167,0.0,0.0235,0.0237,0.0204,0.0207,0.0,0.0,0.0422,0.0118,0.0,0.0312,0.0,0.0,0.0221,0.0,0.0,0.0,0.0,0.0,0.0,0.047,0.0,0.0,0.0663,0.12,0.0,0.0,0.1103,0.0309,0.0,0.0648,0.0,0.0486,0.0,0.0,0.16,0.0,0.0895,0.0091,0.0,0.1875,0.0,0.0945,0.0,0.0,0.0196,0.0,0.0211,0.0676,0.0226,0.0,0.0612,0.0667,0.0,0.0,0.0,0.0,0.0288,0.1316,0.0,0.08,0.0,0.029,0.125,0.0769,0.0898,0.1508,0.0,0.05,0.0,0.0357,0.0,0.0,0.0,0.0,0.2258,0.0,0.0,0.0,0.5,0.2353,0.0,0.0,0.0,0.0,0.0575,0.0,0.0,0.0,0.0401,0.0727,0.1262,0.1387,0.0,0.0,0.0782,0.0,0.0,0.0208,0.1622,0.0,0.0,0.0,0.0,0.0,0.0217,0.0272,0.0,0.0,0.0693,0.1116,0.0142,0.0,0.0625,0.1456,0.0,0.2857,0.1429,0.0,0.0,0.1111,0.0305,0.0244,0.0638,0.0,0.0,0.2356,0.0,0.1613,0.075,0.0693,0.0,0.0443,0.0578,0.0918,0.0,0.0,0.0916,0.0,0.0868,0.0228,0.0,0.0,0.0,0.0,0.0128,0.0,0.0,0.1539,0.0,0.0,0.08,0.0594,0.0,0.0548,0.0594,0.0,0.0,0.0213,0.0787,0.0,0.0581,0.0538,0.0,0.2222,0.0,0.0,0.0,0.2371,1.0,0.018,0.0,0.1835,0.1427,0.0714,0.0561,0.1302,0.0629,0.1098,0.1918,0.0,0.0605,0.0,0.0,0.0,0.1,0.0,0.0435,0.0702,0.0712,0.0,0.0189,0.1113,0.0,0.0,0.0689,0.0,0.0,0.0239,0.1123,0.0,0.0656,0.0646,0.0,0.2526,0.0,0.062,0.0,0.0387,0.0,0.037,0.0,0.0755,0.0756,0.0532,0.122,0.0216,0.0,0.0,0.3333,0.0487,0.0,0.0,0.0,0.0,0.1081,0.0366,0.0167,0.0,0.0,0.0215,0.0,0.0609,0.0811,0.0,0.0,0.0,0.0,0.0972,0.0075,0.0,0.0,0.0625,0.0244,0.0,0.0,0.0,0.0667,0.0,0.0858,0.0,0.0,0.0,0.0,0.0127,0.0,0.0,0.0806,0.0,0.0214,0.0512,0.0712,0.0832,0.0556,0.0,0.1487,0.0791,0.0,0.0476,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.0553,0.0,0.0318,0.0135,0.0,0.1,0.1,0.0689,0.0721,0.0,0.0,0.0,0.1667,0.0857,0.1053,0.05,0.0777,0.0629,0.0266,0.0866,0.04,0.0909,0.1991,0.0,0.0526,0.0634,0.0549,0.0119,0.0245,0.0769,0.0879,0.0,0.125,0.0,0.172,0.0777,0.0,0.0483,0.1178,0.0316,0.1,1.0,0.0,0.0,0.18,0.04,0.0,0.0714,0.0642,0.0,0.0714,0.0207,0.0857,0.0,0.0,0.0,0.0,0.0592,0.1429,0.0319,0.0,0.0198,0.0,0.0,0.035,0.1538,0.0,0.0205,0.0228,0.0,0.0,0.113,0.0,0.0,0.0323,0.0,0.0,0.0,0.0254,0.1656,0.1558,0.0,0.0,0.0,0.0,0.0204,0.1313,0.0,0.2209,0.0323,0.1087,0.0,0.0,0.0,0.0,0.0452,0.0,0.0179,0.0,0.0,0.0,0.0,0.1176,0.0,0.0,0.0645,0.0,0.0,0.0448,0.0833,0.0769,0.0499,0.0,0.0,0.1333,0.0,0.0,0.0056,0.0,0.0,0.0556,0.0571,0.0278,0.0,0.0,0.0,0.0,0.0909,0.0,0.0,0.0792,1.0,0.0,0.0,0.029,0.0,0.0597,0.1364,0.0349,0.0,0.0612,0.0,0.0,0.0232,0.0,0.0593,0.0,0.0577,0.0,0.0,0.0335,0.0575,0.0,0.0,0.0,0.0828,0.0526,0.0,0.0408,0.0354,0.0267,0.0,0.0526,0.0,0.0,0.0,0.0,0.0621,0.045,0.0,0.0,0.0,0.0,0.0,0.0323,0.5,0.0419,0.0,0.0667,0.0784,0.0339,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0476,0.0,0.122,0.0,0.0352,0.0476,0.0,0.0523,0.0,0.0524,0.0,0.0,0.0,0.1429,0.1087,0.0375,0.0,0.0,0.0,0.0,0.0,0.0,0.0292,0.0,0.0,0.0733,0.0,0.0769,0.0568,0.0577,0.0349,0.1346,0.0,0.0,0.0864,0.0533,0.0,0.0,0.0,0.0,0.1038,0.0,0.0,0.0515,0.0652,0.0,0.0333,0.0,0.0,0.1511,0.0641,0.0,0.0,0.0435,0.0,0.0297,0.0469,0.147,0.093,0.0,0.0515,0.0385,0.1111,0.25,0.0,0.1111,0.1154,0.0,0.0,0.0,0.0757,0.0723,0.0,0.0,0.0,0.0281,0.0729,0.0,0.0,0.0,0.1166,0.0,0.0,0.1143,0.0,0.0259,0.0365,0.0,0.0889,0.0476,0.0692,0.0,0.0,0.0,0.0,0.0,0.0,0.0345,0.0,0.0,0.0,0.0,0.0139,0.091,0.0526,0.0515,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0455,0.0,0.0,0.0814,0.0,0.0495,0.0,0.1017,0.0743,0.0363,0.0,0.0,0.0,0.0,0.0,0.079,0.0476,0.1004,0.0181,1.0,0.0,0.0,0.0,0.0552,0.0,0.0634,0.0286,0.0131,0.0435,0.2,0.093,0.0,0.1087,0.1598,0.0,0.0,0.0,0.1667,0.1275,0.0,0.0238,0.0,0.0,0.0,0.0743,0.0398,0.0,0.0315,0.0,0.0,0.0,0.2,0.0,0.2308,0.0909,0.1166,0.0,0.0943,0.0,0.0,0.0,0.0063,0.1204,0.0,0.0,0.0534,0.0,0.1641,0.0659,0.0,0.1113,0.0,0.0645,0.0,0.0,0.2857,0.0,0.0,0.0833,0.0834,0.0,0.0242,0.029,0.0,0.0,0.0588,0.0407,0.0649,0.2,0.0,0.1139,0.0,0.0'


print(f'len {len(lit_cnt_prob_list)}')
def bind_nsml(model,optimizer, task):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        #print('model saved!')
 
    def load(dir_name, *args, **kwargs):
        model.load_weights(os.path.join(dir_name,'model'))
        #print('model loaded')

    def infer(root, phase):
        return _infer(root, phase, model=model,task=task)

    nsml.bind(save=save, load=load, infer=infer)

def _infer(root, phase, model, task):
    # root : csv file path
    # change soon
    #print('_infer root - : ', root)
    #print('_infer phase - : ', phase)

    model, fixlen_feature_names_global, item, image_feature_dict,lit, lit_cnt_prob = get_item(root,phase)
    #bind_nsml(model)
    #bind_nsml(model, [], args.task)
    #print('--get item finished---')
    checkpoint_session = ['3','team_62/airush2/361']
    nsml.load(checkpoint = str(checkpoint_session[0]), session = str(checkpoint_session[1]))
    #print('-- model_load completed --')

    s = time.time()
    data_1_article = item['article_id'].tolist()
    #print('add lit_cnt_prob')
    #print(f'len lit_cnt_prob : {len(lit_cnt_prob)}')
    
    li3 = []
    for i in range(len(data_1_article)):
        if data_1_article[i] not in lit :
            print(data_1_article[i])
            li3.append(0.0)
        else : 
            for j in range(len(lit_cnt_prob)):
                if data_1_article[i] == lit[j] :
                    li3.append(lit_cnt_prob[j])
                    break
        
    item['read_cnt_prob'] = li3

    data_1_article_idxs = item['article_id'].tolist()
    #print(f'data_1_article_idxs[0] : {data_1_article_idxs[0]}')
    li = []

    for i in range(len(data_1_article_idxs)):
        image_feature = image_feature_dict[data_1_article_idxs[i]]
        li.append(image_feature)
    #print('------------is same image picture? let me check---------------')
    #print('article_id : ','757518f4a3da')
    #print('article_if : ',image_feature_dict['757518f4a3da'])
    #print('--------------------------------------------------------------')
    
    item['image_feature'] = li
    li = []
    #print(f'finished data_1_image_feature : {time.time() - s} sec')
    test_generator = data_generator_test(item, fixlen_feature_names_global)
    print(len(item))
    #print(item.head(5))
    # 맞확
    predicts = model.predict_generator(test_generator, steps = len(item), workers = 4)
    #print(f'y_pred shape : {predicts.shape}')
    #print(f'y_pred type : {type(predicts)}')
    #print(predicts)
    predicts = predicts.reshape((len(item),))
    print(predicts.shape)
    pl = predicts.tolist()
    #pl = predicts.tolist()
    #print(pl[:10])
    #print(pl[-10:])
    #print(predicts)
    return pl
    

def data_generator_test(df,fixlen_feature_names_global, batch_size = 1):
    i = 0
    X = df
    while True:
        for i in range(int(np.ceil(len(df)/batch_size))):
            #s = time.time()
            x_batch = X[i*batch_size:(i+1)*batch_size]

            test_model_input = []
            for name in fixlen_feature_names_global:
                #print(name)
                if name == 'image_feature':
                    test_model_input.append(np.array( x_batch['image_feature'].values.tolist()))
                elif name == 'read_cnt_prob':
                    test_model_input.append(x_batch['read_cnt_prob'].values)
                    #print(np.array(x_batch['image_feature'].values.tolist()).shape)
                else:
                    test_model_input.append(x_batch[name + '_onehot'].values)

            yield test_model_input

def scheduler(epoch):
    if epoch < 200:
        return 0.001
    elif epoch < 400 :
        return 0.0005
    elif epoch < 700 :
        return 0.0001
    

class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        print(f"epoch: {epoch}, train_acc: {logs['acc']}")
        nsml.save(str(epoch))

def get_item(root, phase):
    #print('load')
    csv_file = os.path.join(root, 'test', 'test_data', 'test_data')
    item = pd.read_csv(csv_file,
                dtype={
                    'article_id': str,
                    'hh': int, 'gender': str,
                    'age_range': str,
                    'read_article_ids': str
                }, sep='\t')
    #print('loaded!!')
    sparse_features = ['article_id', 'hh','gender','age_range','len_bin']
    dense_features = ['image_feature', 'read_cnt_prob']
        

  
    global lit_cnt_prob_list
    lit_cnt_prob_list = lit_cnt_prob_list.replace(' ','')
    lit_cnt_prob_list = lit_cnt_prob_list.replace('\n','')
    lit_cnt_prob = lit_cnt_prob_list.split(',')


    len_lis = []

    read_article_ids_all = item['read_article_ids'].tolist()
    for i in range(len(item)):
        li = read_article_ids_all[i]
        if type(li) == float:
            len_lis.append(0)
            continue
        len_li = len(li.split(','))
        len_lis.append(len_li)
    
    
    item['len']  = len_lis
    item['len_bin']  = pd.qcut(item['len'],6,duplicates='drop')


    artics = item['article_id'].tolist()
    lit = list(set(artics))
    lit.sort()
    print(f'len lit : {len(lit)}')
    #### fea
    #print('feature dict generate')
    #resnet_feature_extractor('test')

    with open(os.path.join('/data/airush2/test/test_data/test_image_features.pkl'), 'rb') as handle:
        image_feature_dict = pickle.load(handle)
    print('image_feaeture_dict loaded..')
    print('check artic feature')
    print(f"757518f4a3da : {image_feature_dict['757518f4a3da']}")
    
    
    lbe = LabelEncoder()
    lbe.fit(lit)
    item['article_id' + '_onehot'] = lbe.transform(item['article_id'])

    for feat in sparse_features[1:]:
        lbe = LabelEncoder()
        item[feat + '_onehot'] = lbe.fit_transform(item[feat])

    
    #print('----- after onehot encoding -----')
    #print(item.head(10))
    # test set으로 구성해도 되고 item 을..

    fixlen_feature_columns = [SparseFeat('article_id',1896)]
    fixlen_feature_columns += [SparseFeat(feat, item[feat +'_onehot'].nunique()) for feat in sparse_features[1:]]
    fixlen_feature_columns += [DenseFeat('image_feature',len(image_feature_dict[artics[0]]))]
    fixlen_feature_columns += [DenseFeat('read_cnt_prob',1)]
    
    #print(fixlen_feature_columns)
    
    
    idx_artics_all = item['article_id'].tolist()
    
       
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns  
    fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)
    
    fixlen_feature_names_global = fixlen_feature_names

    model = xDeepFM(linear_feature_columns, dnn_feature_columns, task= 'binary')
    #bind_nsml(model, list(), args.task)

    return model, fixlen_feature_names_global, item,image_feature_dict, lit, lit_cnt_prob

import glob
def main(args, local):
    
    if args.arch == 'xDeepFM' and args.mode == 'train':


        s = time.time()
        csv_file = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_data')
        item = pd.read_csv(csv_file,
                    dtype={
                        'article_id': str,
                        'hh': int, 'gender': str,
                        'age_range': str,
                        'read_article_ids': str
                    }, sep='\t')
        label_data_path = os.path.join(DATASET_PATH, 'train',
                                os.path.basename(os.path.normpath(csv_file)).split('_')[0] + '_label')
        label = pd.read_csv(label_data_path,
                    dtype={'label': int},
                    sep='\t')
        item['label']  = label
        s = time.time()
        #print(f'before test article preprocess : {len(item)}')
        
        #print(f'after test  article preprocess : {len(item)}')
        #print(f'time : {time.time() - s}')

        sparse_features = ['article_id', 'hh','gender','age_range','len_bin']
        dense_features = ['image_feature', 'read_cnt_prob']
        target = ['label']
        
        ############################ make more feature !!!!!!! #################################
        ############## 1. read_article_ids len cnt -- user feature #################################################
        len_lis = []

        read_article_ids_all = item['read_article_ids'].tolist()
        for i in range(len(item)):
            li = read_article_ids_all[i]
            if type(li) == float:
                len_lis.append(0)
                continue
            len_li = len(li.split(','))
            len_lis.append(len_li)
        
        
        item['len']  = len_lis
        item['len_bin']  = pd.qcut(item['len'],6,duplicates='drop')
    
        id_to_artic = dict()
        artics = item['article_id'].tolist()
        

        #print(item.head(3))
        #print('columns name : ',item.columns)
        sparse_features = ['article_id', 'hh','gender','age_range','len_bin']
        dense_features = ['image_feature', 'read_cnt_prob']
        
        fixlen_feature_columns = [SparseFeat(feat, item[feat].nunique()) for feat in sparse_features]
        fixlen_feature_columns += [DenseFeat('image_feature',2048)]
        fixlen_feature_columns += [DenseFeat('read_cnt_prob',1)]
        
        #print(f'fixlen_feature_columns : {fixlen_feature_columns}')
 
        
        linear_feature_columns = fixlen_feature_columns
        dnn_feature_columns = fixlen_feature_columns  
        fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)
        print(fixlen_feature_names)
        global fixlen_feature_names_global
        fixlen_feature_names_global = fixlen_feature_names
        model = xDeepFM(linear_feature_columns, dnn_feature_columns, task= 'regression')
        print('---model defined---')
        #print(time.time() - s ,'seconds')


    if use_nsml and args.mode == 'train':

        bind_nsml(model,[], args.task)
    
    
    if args.mode == 'test':
        #print('_infer root - : ', DATASET_PATH)
        #print('test')
        #print('DATASET_PATH: ', DATASET_PATH)
        file_list= glob.glob(f'{DATASET_PATH}/test/test_data/*')
        #print('file_list: ',file_list)
        model, fixlen_feature_names_global, item, image_feature_dict,lit,lit_cnt_prob = get_item(DATASET_PATH,args.mode)
        bind_nsml(model, [], args.task)
        checkpoint_session = ['3','team_62/airush2/361']
        nsml.load(checkpoint = str(checkpoint_session[0]), session = str(checkpoint_session[1])) 
        #print('successfully loaded')

    if (args.mode == 'train'):
        #print('DATASET_PATH: ', DATASET_PATH)
        #file_list= glob.glob(f'{DATASET_PATH}/train/train_data/*')
        #print('file_list :',file_list)
        if args.dry_run:
            print('start dry-running...!')
            args.num_epochs = 1
        else:
            print('start training...!')
        # 미리 전체를 다 만들어놓자 굳이 generator 안써도 되겠네

        nsml.save('infer')
        print('end')
    #print('end_main')

    if args.pause:
        nsml.paused(scope=local)
        #print(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=4)  # not work. check built_in_args in data_local_loader.py

    parser.add_argument('--train_path', type=str, default='train/train_data/train_data')
    parser.add_argument('--test_path', type=str, default='test/test_data/test_data')
    parser.add_argument('--test_tf', type=str, default='[transforms.Resize((456, 232))]')
    parser.add_argument('--train_tf', type=str, default='[transforms.Resize((456, 232))]')

    parser.add_argument('--use_sex', type=bool, default=True)
    parser.add_argument('--use_age', type=bool, default=True)
    parser.add_argument('--use_exposed_time', type=bool, default=True)
    parser.add_argument('--use_read_history', type=bool, default=False)

    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--task', type=str, default='ctrpred')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--save_epoch_every', type=int, default=2)
    parser.add_argument('--save_step_every', type=int, default=1000)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument("--arch", type=str, default="xDeepFM")

    # reserved for nsml
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--iteration", type=str, default='0')
    parser.add_argument("--pause", type=int, default=0)

    parser.add_argument('--dry_run', type=bool, default=False)

    config = parser.parse_args()
    main(config , local = locals())
