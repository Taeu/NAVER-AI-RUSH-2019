import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import pathlib
from model import Baseline, Resnet
import nsml
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataloader import train_dataloader
from dataloader import AIRushDataset


# model load
from octconv import OctResNet, Bottleneck   ### <--- octconv.py 파일에서 필요한 클래스 import
from resnext import *
from model_efficientnet import EfficientNet

#ensemble = [['team_62/airush1/151', '1'],['team_62/airush1/185','17']]
def to_np(t):
    return t.cpu().detach().numpy()

def bind_model(model):
    def save(dir_name, **kwargs):
        save_state_path = os.path.join(dir_name, 'state_dict.pkl')
        state = {
                    'model': model.state_dict(),
                }
        torch.save(state, save_state_path)

    def load(dir_name):
        save_state_path = os.path.join(dir_name, 'state_dict.pkl')
        #print(dir_name)
        state = torch.load(save_state_path)
        model.load_state_dict(state['model'])
        
    
    def infer(test_image_data_path, test_meta_data_path):
        # DONOTCHANGE This Line
        test_meta_data = pd.read_csv(test_meta_data_path, delimiter=',', header=0)
        # dropout ratio  
        # best 였나 이게? 0.4355
        ensemble0 = [['team_62/airush1/320', '02'],['team_62/airush1/320','12'],['team_62/airush1/320','22'],['team_62/airush1/98','4']] # effi
        #ensemble1 = [['team_62/airush1/415', '03'],['team_62/airush1/415','13'],['team_62/airush1/415','23'],['team_62/airush1/415','33']] # effi
        ensemble2 = [['team_62/airush1/678', '02'],['team_62/airush1/678', '12'],['team_62/airush1/185', '17']] #[['team_62/airush1/185','17']] # resnet50
        ensemble3 = [['team_62/airush1/683','02'],['team_62/airush1/409','18']] # oct
        #ensemble4 = [['team_62/airush1/605','8']]  # SKNet # transforms 에서 normalize 반드시 뺄 것
        input_size=224 # you can change this according to your model.
        batch_size=350 # you can change this. But when you use 'nsml submit --test' for test infer, there are only 200 number of data.
        device = 0
        
        w0 = 0.125
        w2 = 0.166
        w3 = 0.25
        w4 = 0.5
        
        predict_list = []
        for i in range(4): # ensemble 개수
            #print('i th inference')
            
            dataloader = DataLoader(
                            AIRushDataset(test_image_data_path, test_meta_data, label_path=None,
                                        transform=transforms.Compose([transforms.Resize((input_size, input_size)),transforms.RandomRotation(20), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
            # 9:10 결과보고 뺄지 말지 결정
            # Let's do ensemble!!!
            if (i == 0):
                # 'efficientNet_b0 : ensemble 4 - fold'
                for j in range(4): 
                    model_name = 'efficientnet-b0'
                    model = EfficientNet.from_name(model_name)
                    bind_model(model)
                    nsml.load(checkpoint=str(ensemble0[j][1]),session=str(ensemble0[j][0]))
                    model.to(device)
                    model.eval()
                    predict_output_list = [] 
                    with torch.no_grad():
                        for batch_idx, image in enumerate(dataloader):
                            image = image.to(device)
                            output = model(image).double()
                            output_prob = to_np(F.softmax(output, dim=1))
                            predict_output_list.append(output_prob * w0)
                    predict_output_list = np.concatenate(predict_output_list,axis=0)
                    predict_list.append(predict_output_list)
            elif (i == 1):
                # resnet50
                for j in range(3):
                    model = resnext50(num_classes=args.output_size) # 모델에 맞게 수정
                    bind_model(model)
                    nsml.load(checkpoint=str(ensemble2[j][1]),session=str(ensemble2[j][0])) # 모델에 맞게 수정
                    model.to(device)
                    model.eval()
                    predict_output_list = [] 
                    with torch.no_grad():
                        for batch_idx, image in enumerate(dataloader):
                            image = image.to(device)
                            output = model(image).double()
                            output_prob = to_np(F.softmax(output, dim=1))
                            #print(output_prob)
                            predict_output_list.append(output_prob * w2)
                    predict_output_list = np.concatenate(predict_output_list,axis=0)
                    predict_list.append(predict_output_list)
                    #print('resnet model')
            elif (i == 2):
                # resnet50
                for j in range(2):
                    model = OctResNet(Bottleneck, [3, 4, 6, 3], num_classes=args.output_size) # 모델에 맞게 수정
                    bind_model(model)
                    nsml.load(checkpoint=str(ensemble3[j][1]),session=str(ensemble3[j][0])) # 모델에 맞게 수정
                    model.to(device)
                    model.eval()
                    predict_output_list = [] 
                    with torch.no_grad():
                        for batch_idx, image in enumerate(dataloader):
                            image = image.to(device)
                            output = model(image).double()
                            output_prob = to_np(F.softmax(output, dim=1))
                            #print(output_prob)
                            predict_output_list.append(output_prob * w3) # 수정
                    predict_output_list = np.concatenate(predict_output_list,axis=0)
                    predict_list.append(predict_output_list)
                    #print('resnet model')
            
            # ensemble 추가

            # 마지막 SENet 추가

        predict_vector = np.argmax(np.sum(predict_list,axis=0), axis=1)
       
        return predict_vector # this return type should be a numpy array which has shape of (138343)

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser')
    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--pause', type=int, default=0)
    
    # custom args
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu_num', type=int, nargs='+', default=[0])
    parser.add_argument('--resnet', default=True)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--output_size', type=int, default=350) # Fixed
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    if args.resnet:
        assert args.input_size == 224
        
        model_name = 'efficientnet-b0'
        model = EfficientNet.from_name(model_name)
    
    else:
        model = Baseline(args.hidden_size, args.output_size)
    
    bind_model(model)
    if args.mode == "train":
        
        nsml.save('E')
        print('---end---')

    if args.pause:
        nsml.paused(scope=locals())
    
    
