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


if not nsml.IS_ON_NSML:
    DATASET_PATH = os.path.join('/airush2_temp')
    DATASET_NAME = 'airush2_temp'
    print('use local gpu...!')
    use_nsml = False
else:
    DATASET_PATH = os.path.join(nsml.DATASET_PATH)
    print('start using nsml...!')
    print('DATASET_PATH: ', DATASET_PATH)
    use_nsml = True


fixlen_feature_names_global=[]
test_article_list = """77e5b6e039c2 ,
    021c7a9a7bce ,
    8890cb1dfefa ,
    0c1802e5e937 ,
    976cff979cb5 ,
    bdcae37688d6 ,
    9a8b3b8b2f1a ,
    80f1d8c0296c ,
    b0b547b7ac11 ,
    0c35696f126f ,
    8e9fdc2e166f ,
    aa6abe62774c ,
    faf1a4d1d15b ,
    23a1c88b5f99 ,
    bb526f9d97f4 ,
    09a718882306 ,
    37a2d827e370 ,
    1d9fcffd2aee ,
    f7f49530f2ab ,
    b8350e3a0ff4 ,
    a6e3bc5696e7 ,
    4309a7e0dfdd ,
    567921d2e080 ,
    ec2554421279 ,
    4918b9e5fcaf ,
    9de5e64e7c39 ,
    d28ceaaccd35 ,
    057fa8d9c5a3 ,
    666bb50bb190 ,
    45c96f112621 ,
    2e6c29a3ee93 ,
    4144d5df324d ,
    f158892a8f01 ,
    a1346fa61a57 ,
    ebba9f2954a8 ,
    9271c65ba187 ,
    4bbc235c85f0 ,
    1e950292d51c ,
    ec5ba3fafe06 ,
    eee6750dfa61 ,
    ee87bf88348d ,
    b8f850c522d7 ,
    aeb307997f9f ,
    5b0e4e70d9cf ,
    6b9ab6e9f5a4 ,
    3dd7f7b7a2ac ,
    ebad6a3e2b95 ,
    e946906378dc ,
    8cb7760e77fa ,
    e9394de05221 ,
    d44c46d5ecbd ,
    e851f004d7bf ,
    bee1965229fd ,
    333a595b31f8 ,
    70aa42a6afb2 ,
    6a2ea703c00a ,
    278e94c9b7be ,
    d45c9da5467d ,
    caeb470c7b4b ,
    a31ff2b18124 ,
    39c0d9cc071b ,
    9a0ba678dd77 ,
    7e5b723fe3c9 ,
    f34a1f1f1abf ,
    d9cc8a65467b ,
    2bd69ba7969a ,
    3430ff15798e ,
    fd10201f874a ,
    0d20a6b11ecd ,
    18d84009afda ,
    24d1d464115d ,
    af3700430574 ,
    4fb96586805d ,
    c2fa320b527e ,
    510f411fdfa4 ,
    69123a97dbe9 ,
    e5eddc038dd4 ,
    23ed7f849a13 ,
    36d9adb79d3b ,
    91372e8363a2 ,
    cc96d794d74d ,
    4b418aa37766 ,
    739f709ec856 ,
    7655305189aa ,
    8ef7d86b1305 ,
    cce761f22272 ,
    aec071ad5d97 ,
    a0342047a43f ,
    feefc86ee1bf ,
    9e22a83b8f68 ,
    a3a70d265197 ,
    279a41002361 ,
    0caeac13b80f ,
    90d6f8fabe1c ,
    73962f0bf6b5 ,
    51d345549c29 ,
    9aea4a4fcb41 ,
    51b3d6885a43 ,
    62df4af473e6 ,
    a785ad42d358 ,
    d896527761d8 ,
    1bb9af645346 ,
    f368aadd41e0 ,
    f9c2a34df0f8 ,
    2b1c4af99f9c ,
    9a1244d0da42 ,
    a81032ebb75c ,
    0761b79d3a7e ,
    413275053602 ,
    655e5b424b5a ,
    8a37033e9778 ,
    040df7d04d83 ,
    41e946fbd803 ,
    8895fcfa5dca ,
    5d6644e7abda ,
    d4cf43815138 ,
    b6e9a82209ef ,
    0b3dcf6fe3c0 ,
    4d45bdaeaba8 ,
    0f8fe6b57e23 ,
    95e5558455cf ,
    8e437a20d995 ,
    19d87bb869d5 ,
    4add8bf37f0a ,
    2876ea06913a ,
    60a3aec22f15 ,
    3d5cf7d1c8e7 ,
    c52483759825 ,
    7a6ba503c717 ,
    8ed00aeed4cd ,
    10bb1e1bde7c ,
    4c5d7b2d3d56 ,
    9a496af61128 ,
    a3b94960df5d ,
    f8fbacbccb70 ,
    562cc6e961d0 ,
    1c2ba8adc698 ,
    048c92d99066 ,
    760f22e7eeea ,
    e738fcea8a05 ,
    a6956ca895bb ,
    1ddf4410af6d ,
    9245d4db3410 ,
    00966b684191 ,
    212095594257 ,
    da87ea39f000 ,
    dec374e53bef ,
    95fb849ed9b6 ,
    e40c29049121 ,
    ee61b00f1244 ,
    be4397b5f9f1 ,
    4a6620e1c32c ,
    006243b07ca1 ,
    7d86c289dda9 ,
    3d10c6864aa5 ,
    8cc3f1a70740 ,
    ff7eb712d4f5 ,
    06b1bafdc206 ,
    83b820e42af4 ,
    a2e16a6693aa ,
    b236381a17be ,
    7ce353094cce ,
    15f3566d639a ,
    eaafa4695786 ,
    5434de80e9f4 ,
    679329164024 ,
    b3b4a528a019 ,
    8ad228511267 ,
    feac192d9dab ,
    f52b69bbd37b ,
    63845948aac4 ,
    4a822d4b6eb6 ,
    483a99c0d4b1 ,
    52c77aa0fb00 ,
    717ef5b8a3b4 ,
    a46306110292 ,
    09a71c94831d ,
    b41f6b968104 ,
    bfde563f2df4 ,
    2e7d254f45ec ,
    c40d07a7e254 ,
    7510d62fc747 ,
    d8761ca60920 ,
    2d0d3d0bd826 ,
    2cda4aeef377 ,
    f4d380732f5f ,
    489456a45e15 ,
    f17e17a317e8 ,
    bfa1431e35d4 ,
    cc8d397a85ad ,
    471c5419e4e8 ,
    7d3c888019f6 ,
    3f85bf30834d ,
    8b0ff466ddf4 ,
    0e3798b2f06b ,
    ae9d091062c3 ,
    7a47e6459638 ,
    cb43230d2f03 ,
    1a87662bf8a5 ,
    f429dadfec02 ,
    172100d1cd98 ,
    e89a47ee22fa ,
    a1c0e8271b5d ,
    5aff94ad02a4 ,
    7cf2cd421cdc ,
    757518f4a3da ,
    bca87fa19575 ,
    8d3ac9743280 ,
    6c2e79e49504 ,
    3385da5e04e2 ,
    167f1200b7ea ,
    8fb1d4b8a472 ,
    c1e4ef82c923 ,
    9dc8b827fc54 ,
    0dc85c4c443e ,
    7d31dd8fe8f0 ,
    96e8fb7c9d84 ,
    219bd5492d88 ,
    b60f5c8816ff ,
    ea4f4679096b ,
    4182eabd0476 ,
    a5e218237de4 ,
    c50010088bc5 ,
    f6701d521020 ,
    3b9a91c7168c ,
    55c7d7620686 ,
    570a4863ff5c ,
    66c56d29e646 ,
    28717c591598 ,
    17e8499a0201 ,
    7fc6f6fa444c ,
    bdccf137f4de ,
    958a2f67dece ,
    c6333ca678cf ,
    ad9bb0c5d192 ,
    8fd83920de7d ,
    78b1d0b048c0 ,
    ac1ff3dd003c ,
    1e26fb94d125 ,
    539e3815b099 ,
    81677b23a8bf ,
    52020d1b9eff ,
    976ee5388184 ,
    ff86e8af08fb ,
    e97ed8836f58 ,
    521d29334e1f ,
    e5c755526464 ,
    91a97896af8e ,
    17de6b5cb2e4 ,
    220aa4537563 ,
    bcd925f2451c ,
    92aa12f62d7a ,
    9d8ef57b6fe7 ,
    a1b77e768e49 ,
    fd4199a48fc0 ,
    10f05eea8c30 ,
    397ac7fbccb3 ,
    698b8aa17fa2 ,
    90b6252ef8af ,
    8c0d61d2e071 ,
    5eb1604a611d ,
    daa43e17ee7b ,
    5945b35613d0 ,
    8bdb78f20404 ,
    489facee3880 ,
    dbb64cd751c3 ,
    1184176f0013 ,
    3ed0728317a3 ,
    8e2352b92634 ,
    e14431c72939 ,
    f73072810050 ,
    f9a72642b160 ,
    eee9c1f4222b ,
    78778bae32f4 ,
    06425238a5e2 ,
    c52db15596d5 ,
    6119e80a9d76 ,
    ad90521bad85 ,
    7cdbfed4fcf6 ,
    432c68ed2dac ,
    f7e9bd26aca0 ,
    978f08f206e5 ,
    ddb106717dca ,
    01e234fed982 ,
    0e74c0267efe ,
    8c63cc7588c9 ,
    7dfad497f5b7 ,
    72c6a88c3552 ,
    df9ae04ddbab ,
    562104faadf4 ,
    973bfa361f12 ,
    0df49fd3dcf1 ,
    782cf9107650 ,
    b9aa157a5c28 ,
    99331857f173 ,
    98cf6695af2a ,
    c965c3c5bc56 ,
    6d42b1067dc9 ,
    fce053b43bd7 ,
    6eca9900ec18 ,
    bce18bc378b5 ,
    c550a4df4751 ,
    8cebb42ffb63 ,
    0e9b690eaaad ,
    07d3ab4cc962 ,
    972e4e986548 ,
    6db22793453e ,
    c14724f8af97 ,
    a5d21b02a3ce ,
    46213cbac7fd ,
    309d2aa34f18 ,
    3d25b9caa11b ,
    273734d766c2 ,
    bc7d6a9f35bc ,
    06a6828d8873 ,
    0e844f4b0260 ,
    5ef506c546dc ,
    cea063a3f639 ,
    939c24c1c092 ,
    6a24a7ac9174 ,
    b96b2e0f25a1 ,
    1dffee8e7adc ,
    05d2a03b6fc4 ,
    861462181b3e ,
    be8576cfc7ad ,
    6cf3e3170774 ,
    bb073da7979e ,
    1cfd62a4588a ,
    0b14ba8b524f ,
    1200ccaaa954 ,
    e5599fcbc888 ,
    1be38e6b0cb0 ,
    8fc9dbf4dc28 ,
    668b743836ab ,
    211c50b68c51 ,
    404d8787e99c ,
    974f572a7bec ,
    93a3786c8f6e ,
    ea8521b697ee ,
    a8a351d97392 ,
    6a7ad2dbca7d ,
    bb29e1a2e499 ,
    b20683449899 ,
    b2a59e532589 ,
    b65025d4d71c ,
    18f078d34154 ,
    6ff5ef30f627 ,
    8fe227977fd0 ,
    89c9f419ad9c ,
    29c81dc013f8 ,
    17c92caaea73 ,
    5b73f5efb3ec ,
    b43c5d6e0eb9 ,
    b67cdaaf5da6 ,
    414fe97ba590 ,
    dc0af193f61c ,
    e7c01c8e3382 ,
    d59b84e95c78 ,
    b19f2364f219 ,
    6e88169ee3fd ,
    bb730a593c2e ,
    106d9b22a875 ,
    df02f4495a78 ,
    e88b161e0aa1 ,
    4d7eb479defa ,
    2e487d73c82f ,
    2f8e9f3112ac ,
    0c4d0ed64adc ,
    945780210a9f ,
    df1d6bffa0ee ,
    0ac5cb215acd ,
    99b10e9580c6 ,
    3b006fc4fe6c ,
    acb153845a8c ,
    1c06c901e7e9 ,
    35eae24dd04b ,
    df925261ffc2 ,
    14888f314294 ,
    2f6139c6b61e ,
    2c54f7d416ba ,
    3b91909a89a5 ,
    2f0ec6618455 ,
    1bfae940e040 ,
    29143a9b1dd2 ,
    2528d973d976 ,
    c19b4da61b0e ,
    8b545e858bed ,
    59e8fc918d0e ,
    80774a8069f1 ,
    c5e5b20b2e15 ,
    08a2af512c10 ,
    6dd5f60bbc51 ,
    9ac1d03ee1b4 ,
    4127e0857764 ,
    d0b90d2ea589 ,
    06516f9f1d2d ,
    83d81fd39e33 ,
    60786ea2b79c ,
    cd5d0d69d098 ,
    1d202f43cb05 ,
    bf23f380c019 ,
    ecf3e4e042fb ,
    f35bbe6a36c9 ,
    7822ac70113e ,
    448f27be7ec0 ,
    68040e12bfa5 ,
    7161935cffaa ,
    de5115154e05 ,
    718b277d3f96 ,
    94bc04847bd7 ,
    f9b61bc2330c ,
    2801b54b20d7 ,
    ac83e8b41a4d ,
    dcb2dac12fa3 ,
    643380adecf2 ,
    3665ca67d16d ,
    1817dec6ed5d ,
    795b04ffc210 ,
    060f32817dc1 ,
    0a5c515306b9 ,
    c194e8fcb414 ,
    eb0897ff63b7 ,
    c846ab81284e ,
    555904c20672 ,
    981eee0a340c ,
    f2dfd9c704af ,
    4b6e0d22163d ,
    d0712e5c29e3 ,
    b6fb269aba2c ,
    3f75b9524cd6 ,
    487449891028 ,
    ded32a54a9b6 ,
    ba66be89b149 ,
    5dc08dca028c ,
    93bf5ed88ff7 ,
    ebb944878b3b ,
    c85b32ff0ea0 ,
    82f2ca897bc0 ,
    5b6d1e9b38ce ,
    c97468592027 ,
    8cb3bfd3e0fa ,
    a5631f055728 ,
    125453c9b779 ,
    300b4e376c54 ,
    f0279a582893 ,
    a60b8207c44e ,
    a3d13f08c16f ,
    66542bdce394 ,
    878afdca49ce ,
    65bd390c6f35 ,
    b906f6e9e122 ,
    c09a4a75675d ,
    ed173d87cf27 ,
    d66563a074dc ,
    38e53b9aacdb ,
    b3b4dab25327 ,
    c5621834477f ,
    94dab122b11f ,
    8a990023c16a ,
    5887a9b474d8 ,
    ef7e4793fc0a ,
    7f89891db083 ,
    1b8043d3cfca ,
    2b1ecd1d3823 ,
    6e4020c1b55a ,
    80ad213fed16 ,
    135d237dcdc7 ,
    a5b8262d20d5 ,
    edb4d09eddde ,
    5fd7543690dd ,
    9fe54018590c ,
    7d3d6ab7740d ,
    b6359ddd5474 ,
    45e19ca03a21 ,
    1c6cd986daf5 ,
    cb2633cad640 ,
    b59a50096000 ,
    810dff385538 ,
    bd598ca72f43 ,
    aaa28a3b57e8 ,
    b7aeedc15982 ,
    33d56bfae208 ,
    ac34843ffe0e ,
    3cf359f05649 ,
    b8e8f6969b69 ,
    64f606bb9359 ,
    a9b71e96bf0e ,
    d237df0f1673 ,
    dd8b0059c565 ,
    619d03aae8bd ,
    2dce71a5e251 ,
    7c0b90fcb4fc ,
    cc62d0aa4f16 ,
    986734f88e74 ,
    7011bfe7cb22 ,
    00d31669d80c ,
    06163a992c49 ,
    aa412196258a ,
    57cff4edcb74 ,
    fcd6984032a8 ,
    cbfce83c0d4a ,
    c36b55052c0c ,
    358557698b7c ,
    870b53a4518a ,
    97dee3e21af1 ,
    3d55221e6dc2 ,
    e9bb3122a0b0 ,
    1b781f98d007 ,
    3da5f1fd352c ,
    8c4f48a292a0 ,
    cf1a38026f11 ,
    308c474e5f2c ,
    086bcc4c803d ,
    0ce15d61f967 ,
    6d10bdb0442c ,
    666439a883f5 ,
    edb0e0dcfc0c ,
    4f2e4a92074b ,
    dc84af9515d8 ,
    38828a108ee4 ,
    135b2c0af605"""

# bind
def bind_nsml(model, optimizer, task):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')
 
    def load(dir_name, *args, **kwargs):
        model.load_weights(os.path.join(dir_name,'model'))
        print('model loaded')

    def infer(root, phase):
        return _infer(root, phase, model=model, task=task)

    nsml.bind(save=save, load=load, infer=infer)

def _infer(root, phase, model, task):

    print('_infer root - : ', root) 
    y_pred = []
        
    return y_pred

def data_generator(df, batch_size = 2048):
    i = 0
    Y = df['label']
    X = df.drop(['label'],axis = 1)
    while True:
        for i in range(int(np.ceil(len(df)/batch_size))):
            #s = time.time()
            x_batch = X[i*batch_size:(i+1)*batch_size]
            y_batch = np.array(Y[i*batch_size:(i+1)*batch_size]).astype('float32')


            #train, test = train_test_split(x_batch, test_size=0.2,shuffle=False)
            #train_label, test_label = train_test_split(y_batch, test_size=0.2,shuffle=False)
            train_model_input = []
            for name in fixlen_feature_names_global:
                #print(name)
                if name == 'image_feature':
                    train_model_input.append(np.array( x_batch['image_feature'].values.tolist()))
                    #print(np.array(x_batch['image_feature'].values.tolist()).shape)
                else:
                    train_model_input.append(x_batch[name +'_onehot'].values)

            
            #print(f'len train_model_input {len(train_model_input)}')
            #print(f'len train_label {len(y_batch)}')
            #print(f'generated batch {time.time() - s} sec')

            yield train_model_input, y_batch

def scheduler(epoch):
    if epoch < 2:
        return 0.001
    elif epoch < 4 :
        return 0.0001
    else:
        return 0.00001
    
class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        print(f"epoch: {epoch}, train_acc: {logs['acc']}")
        nsml.save(str(epoch))


def main(args):
    
    if args.arch == 'xDeepFM':
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
        print(f'before test article preprocess : {len(item)}')
        global test_article_list
        test_article_list = test_article_list.replace(' ','')
        test_article_list = test_article_list.replace('\n','')
        lit = test_article_list.split(',')
        item = item[item['article_id'].isin(lit)]
        
        print(f'after test  article preprocess : {len(item)}')
        print(f'time : {time.time() - s}')

        sparse_features = ['article_id', 'hh','gender','age_range','len_bin']
        dense_features = ['image_feature']
        target = ['label']
        
        
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
        
        read_cnt = item[item['label'] == 1].groupby('article_id').agg({'hh' : 'count'})
        read_cnt = read_cnt.reset_index()
        read_cnt = read_cnt.rename(columns = {'hh':'read_cnt'})

        read_cnt_list = read_cnt['read_cnt'].tolist()
        read_cnt_artic_list = read_cnt['article_id'].tolist()
        print(f'len read_cnt : {len(read_cnt)}')
        print(read_cnt.head(3))
        
        total_cnt = item.groupby('article_id').agg({'hh' : 'count'})
        total_cnt = total_cnt.reset_index()
        total_cnt = total_cnt.rename(columns = {'hh':'read_cnt'})
        total_cnt_list = total_cnt['read_cnt'].tolist()
        total_cnt_artic_list = total_cnt['article_id'].tolist()
        print(f'len read_cnt : {len(total_cnt)}')
        print(total_cnt.head(3))



        # One hot Encoding
        with open(os.path.join(DATASET_PATH, 'train', 'train_data', 'train_image_features.pkl'), 'rb') as handle:
            image_feature_dict = pickle.load(handle)
        for feat in sparse_features:
            lbe = LabelEncoder()
            item[feat +'_onehot'] = lbe.fit_transform(item[feat]) # 이때 고친 라벨이 같은 라벨인지도 필수로 확인해야함

        print(item.head(10))
        print('columns name : ',item.columns)
        fixlen_feature_columns = [SparseFeat(feat, item[feat +'_onehot'].nunique()) for feat in sparse_features]
        fixlen_feature_columns += [DenseFeat(feat,len(image_feature_dict[artics[0]])) for feat in dense_features]
        
        idx_artics_all = item['article_id'+'_onehot'].tolist()
        
        for i in range(len(artics)):
            idx_artic = idx_artics_all[i]
            if idx_artic not in id_to_artic.keys():
                id_to_artic[idx_artic] = artics[i]
        
        linear_feature_columns = fixlen_feature_columns
        dnn_feature_columns = fixlen_feature_columns  
        fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)
        print(fixlen_feature_names)
        global fixlen_feature_names_global
        fixlen_feature_names_global = fixlen_feature_names
        model = xDeepFM(linear_feature_columns, dnn_feature_columns, task= 'regression')
        print('---model defined---')
        print(time.time() - s ,'seconds')

    optimizer = tf.keras.optimizers.Adam(args.lr)
    s = time.time()

    # negative sampling
    item_pos = item[item['label'] == 1] 
    item_neg = item[item['label'] == 0]
    

    dn_1 = item_neg.sample(n=3*len(item_pos), random_state=42)
    dn_2 = item_neg.sample(n=3*len(item_pos), random_state=20)
    dn_3 = item_neg.sample(n=3*len(item_pos), random_state=7)
    dn_4 = item_neg.sample(n=3*len(item_pos), random_state=33)
    dn_5 = item_neg.sample(n=3*len(item_pos), random_state=41)

    dn_1.reset_index()
    
    data_1 = pd.concat([dn_1,item_pos]).sample(frac=1, random_state=42).reset_index()
    print(f'len data_1 : {len(data_1)}')
    print(data_1.head(5))
    data_2 = pd.concat([dn_2,item_pos]).sample(frac=1, random_state=42).reset_index()
    data_3 = pd.concat([dn_3,item_pos]).sample(frac=1, random_state=42).reset_index()
    data_4 = pd.concat([dn_4,item_pos]).sample(frac=1, random_state=42).reset_index()
    data_5 = pd.concat([dn_5,item_pos]).sample(frac=1, random_state=42).reset_index()

    data_1_article_idxs = data_1['article_id_onehot'].tolist()
    data_1_article = data_1['article_id'].tolist()
    li = []
    for i in range(len(data_1_article_idxs)):
        image_feature = image_feature_dict[id_to_artic[data_1_article_idxs[i]]]
        li.append(image_feature)
    print(f'article_id : {data_1_article[0]}')
    print(f'article_image_feature : {image_feature_dict[data_1_article[0]]}')
    
    data_1['image_feature'] = li
    li = []
    print(f'finished data_1_image_feature : {time.time() - s} sec')



    if use_nsml:
        bind_nsml(model, optimizer, args.task)
    if args.pause:
        nsml.paused(scope=locals())

    if (args.mode == 'train') or args.dry_run:
        best_loss = 1000
        if args.dry_run:
            print('start dry-running...!')
            args.num_epochs = 1
        else:
            print('start training...!')
        # 미리 전체를 다 만들어놓자 굳이 generator 안써도 되겠네
        model.compile(tf.keras.optimizers.Adam(args.lr),'mse',metrics=['accuracy'],)
        train_generator = data_generator(data_1)
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
        
        save_cbk = CustomModelCheckpoint()

        history = model.fit_generator(train_generator,
                     epochs=10, verbose=2, workers = 8, steps_per_epoch=np.ceil(len(data_1)/2048), callbacks = [lr_scheduler,save_cbk])
        print('again')
        


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
    main(config)
