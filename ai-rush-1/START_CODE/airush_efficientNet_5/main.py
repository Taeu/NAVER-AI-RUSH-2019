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
from model_efficientnet import EfficientNet
from torch.optim.lr_scheduler import StepLR
#from torchsummary import summary

def to_np(t):
    return t.cpu().detach().numpy()

def bind_model(model_nsml):
    def save(dir_name, **kwargs):
        save_state_path = os.path.join(dir_name, 'state_dict.pkl')
        state = {
                    'model': model_nsml.state_dict(),
                }
        torch.save(state, save_state_path)

    def load(dir_name):
        save_state_path = os.path.join(dir_name, 'state_dict.pkl')
        state = torch.load(save_state_path)
        model_nsml.load_state_dict(state['model'])
        
    def infer(test_image_data_path, test_meta_data_path):
        # DONOTCHANGE This Line
        test_meta_data = pd.read_csv(test_meta_data_path, delimiter=',', header=0)
        
        input_size=224 # you can change this according to your model.
        batch_size=100 # you can change this. But when you use 'nsml submit --test' for test infer, there are only 200 number of data.
        device = 0
        
        dataloader = DataLoader(
                        AIRushDataset(test_image_data_path, test_meta_data, label_path=None,
                                      transform=transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])),
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True)
        
        model_nsml.to(device)
        model_nsml.eval()
        predict_list = []
        for batch_idx, image in enumerate(dataloader):
            image = image.to(device)
            output = model_nsml(image).double()
            
            output_prob = F.softmax(output, dim=1)
            predict = np.argmax(to_np(output_prob), axis=1)
            predict_list.append(predict)
                
        predict_vector = np.concatenate(predict_list, axis=0)
        return predict_vector # this return type should be a numpy array which has shape of (138343)

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)

# dropout ratio  : 0.3 ì¼ë¡ ì¡°ì 
    
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
    parser.add_argument('--epochs', type=int, default=10) # change
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4) # change
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    if args.resnet:
        assert args.input_size == 224
        #model = Resnet(args.output_size)
        print('!!!!!!!!!!!!!!!!efficientnet load!!!!!!!!!!!!!!!!')
        model_name = 'efficientnet-b0'
        print(model_name)
        
        model = EfficientNet.from_name(model_name)
        
        #model = EfficientNet.from_pretrained(model_name, num_classes=350)
        #summary(model,input_size=(3,224,224))
    else:
        model = Baseline(args.hidden_size, args.output_size)
    optimizer = optim.Adam(model.parameters(), args.learning_rate)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = nn.CrossEntropyLoss() #multi-class classification task

    model = model.to(device)
    model.train()

    # DONOTCHANGE: They are reserved for nsml
    bind_model(model)
    # below the nsml load
    nsml.load(checkpoint='3',session='team_62/airush1/98')
    nsml.save('T')


    if args.pause:
        nsml.paused(scope=locals())
    if args.mode == "train":
        # Warning: Do not load data before this line
        dataloader = train_dataloader(args.input_size, args.batch_size, args.num_workers)

        for epoch_idx in range(1, args.epochs + 1):
            scheduler.step()
            print('Epoch:', epoch_idx,'LR:', scheduler.get_lr())

            total_loss = 0
            total_correct = 0
            cnt_valid = 0
            total_correct_valid = 0
            batch_len = 0
            for batch_idx, (image, tags) in enumerate(dataloader):
                batch_len += args.batch_size
                if batch_idx >= len(dataloader) - 1 :
                    print(batch_idx)
                    break
                optimizer.zero_grad()

                image = image.to(device) #torch.Size([batch_size, 3, 224, 224])
                image_train = image[:int(args.batch_size*0.9),:,:,:]
                image_valid = image[int(args.batch_size*0.9):,:,:,:]


                tags = tags.to(device) #torch.Size([64])
                tags_train = tags[:int(args.batch_size*0.9)]
                tags_valid = tags[int(args.batch_size*0.9):]
                
                 
                output_valid = model(image_valid).double()
                output_prob_valid = F.softmax(output_valid, dim=1) #softmaxë êµ³ì´ íììê³ 
                predict_vector_valid = np.argmax(to_np(output_prob_valid), axis=1)
                label_vector_valid = to_np(tags_valid)
                bool_vector_valid = predict_vector_valid == label_vector_valid
                accuracy_valid = bool_vector_valid.sum() / len(bool_vector_valid)



                output_train = model(image_train).double() # torch.Size([batch_size, 350])
                

                loss = criterion(output_train, tags_train) # criterion ì ë¤ì íì¸íì
                loss.backward()
                optimizer.step()

                output_prob_train = F.softmax(output_train, dim=1) #softmaxë êµ³ì´ íììê³ 
                predict_vector_train = np.argmax(to_np(output_prob_train), axis=1)
                label_vector_train = to_np(tags_train)
                bool_vector_train = predict_vector_train == label_vector_train
                accuracy = bool_vector_train.sum() / len(bool_vector_train)





                total_loss += loss.item()
                total_correct += bool_vector_train.sum()
                total_correct_valid += bool_vector_valid.sum()

                if batch_idx % args.log_interval == 0:
                    cnt_valid += 1
                    
                    print('Batch {} / {}: Batch Loss {:2.4f} / Batch Acc {:2.4f}'.format(batch_idx,
                                                                             len(dataloader),
                                                                             total_loss/(batch_len*0.9),
                                                                             total_correct/(batch_len*0.9)))
                    print('valid {} / {}: Valid Acc {:2.4f}'.format(batch_idx,len(dataloader),total_correct_valid/(batch_len*0.1)))

                    

     
            nsml.save(epoch_idx)
            print('Epoch {} / {}: Loss {:2.4f} / Epoch Acc {:2.4f}'.format(epoch_idx,
                                                           args.epochs,
                                                           total_loss/len(dataloader.dataset),
                                                           total_correct/len(dataloader.dataset)))
            nsml.report(
                summary=True,
                step=epoch_idx,
                scope=locals(),
                **{
                "train__Loss": total_loss/len(dataloader.dataset),
                "train__Accuracy": total_correct/(len(dataloader.dataset)*0.9),
                "valid__Accuracy": total_correct_valid /(len(dataloader.dataset) *0.1),
                })
