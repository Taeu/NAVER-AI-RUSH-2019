from data_local_loader import get_data_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import argparse
import numpy as np
import time
import datetime

from data_loader import feed_infer
from evaluation import evaluation_metrics
import nsml

# expected to be a difficult problem
# Gives other meta data (gender age, etc.) but it's hard to predict click through rate
# How to use image and search history seems to be the key to problem solving. Very important data
# Image processing is key. hint: A unique image can be much smaller than the number of data.
# For example, storing image features separately and stacking them first,
# then reading them and learning artificial neural networks is good in terms of GPU efficiency.
# -> image feature has been extracted and loaded separately.
# The retrieval history is how to preprocess the sequential data and train it on which model.
# Greatly needed efficient coding of CNN RNNs.
# You can also try to change the training data set itself. Because it deals with very imbalanced problems.
# Refactor to summarize from existing experiment code.

if not nsml.IS_ON_NSML:
    DATASET_PATH = os.path.join('/home/kwpark_mk2/airush2_temp')
    DATASET_NAME = 'airush2_temp'
    print('use local gpu...!')
    use_nsml = False
else:
    DATASET_PATH = os.path.join(nsml.DATASET_PATH)
    print('start using nsml...!')
    print('DATASET_PATH: ', DATASET_PATH)
    use_nsml = True


# example model and code
class MLP_only_flatfeatures(nn.Module):
    def __init__(self, num_classes=1):
        super(MLP_only_flatfeatures, self).__init__()
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(2083, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.num_classes),
        )

        self._initialize_weights()

    def forward(self, extracted_image_feature, flat_features):
        x = torch.cat((extracted_image_feature, flat_features), 1)
        x = self.classifier(x)
        # x = self.relu(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# example model and code
class CTRResNet_CAT(models.ResNet):
    def __init__(self, block, layers, num_classes=1):
        super().__init__(block, layers, num_classes=num_classes)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Linear(547, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x, flat_features):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # concat, hint: how you are going to fuse these features..?
        x = x.view(x.size(0), -1)
        x = torch.cat((x, flat_features), 1)

        x = self.classifier(x)
        return x


def get_mlp(num_classes):
    return MLP_only_flatfeatures(num_classes=num_classes)


def get_resnet18(num_classes):
    return CTRResNet_CAT(models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def bind_nsml(model, optimizer, task):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(dir_name, 'model.ckpt'))
        print('saved model checkpoints...!')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.ckpt'))
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print('loaded model checkpoints...!')

    def infer(root, phase):
        return _infer(root, phase, model=model, task=task)

    nsml.bind(save=save, load=load, infer=infer)


def _infer(root, phase, model, task):
    # root : csv file path
    print('_infer root - : ', root)
    with torch.no_grad():
        model.eval()
        test_loader, dataset_sizes = get_data_loader(root, phase)
        y_pred = []
        print('start infer')
        for i, data in enumerate(test_loader):
            images, extracted_image_features, labels, flat_features = data

            # images = images.cuda()
            extracted_image_features = extracted_image_features.cuda()
            flat_features = flat_features.cuda()
            # labels = labels.cuda()

            logits = model(extracted_image_features, flat_features)
            y_pred += logits.cpu().squeeze().numpy().tolist()

        print('end infer')
    return y_pred


def main(args):
    if args.arch == 'MLP':
        model = get_mlp(num_classes=args.num_classes)
    elif args.arch == 'Resnet':
        model = get_resnet18(num_classes=args.num_classes)

    if args.use_gpu:
        model = model.cuda()
    else:
        model = model.cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if use_nsml:
        bind_nsml(model, optimizer, args.task)
    if args.pause:
        nsml.paused(scope=locals())

    if (args.mode == 'train') or args.dry_run:
        train_loader, dataset_sizes = get_data_loader(
            root=os.path.join(DATASET_PATH, 'train', 'train_data', 'train_data'),
            phase='train',
            batch_size=args.batch_size)

        start_time = datetime.datetime.now()
        iter_per_epoch = len(train_loader)
        best_loss = 1000
        if args.dry_run:
            print('start dry-running...!')
            args.num_epochs = 1
        else:
            print('start training...!')

        for epoch in range(args.num_epochs):
            for i, data in enumerate(train_loader):
                images, extracted_image_features, labels, flat_features = data

                images = images.cuda()
                extracted_image_features = extracted_image_features.cuda()
                flat_features = flat_features.cuda()
                labels = labels.cuda()

                # forward
                if args.arch == 'MLP':
                    logits = model(extracted_image_features, flat_features)
                elif args.arch == 'Resnet':
                    logits = model(images, flat_features)
                criterion = nn.MSELoss()
                loss = torch.sqrt(criterion(logits.squeeze(), labels.float()))

                # backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if loss < best_loss:
                    nsml.save('best_loss')  # this will save your best model on nsml.

                if i % args.print_every == 0:
                    elapsed = datetime.datetime.now() - start_time
                    print('Elapsed [%s], Epoch [%i/%i], Step [%i/%i], Loss: %.4f'
                          % (elapsed, epoch + 1, args.num_epochs, i + 1, iter_per_epoch, loss.item()))
                if i % args.save_step_every == 0:
                    # print('debug ] save testing purpose')
                    nsml.save('step_' + str(i))  # this will save your current model on nsml.
            if epoch % args.save_epoch_every == 0:
                nsml.save('epoch_' + str(epoch))  # this will save your current model on nsml.
    nsml.save('final')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=0)  # not work. check built_in_args in data_local_loader.py

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
    parser.add_argument("--arch", type=str, default="MLP")

    # reserved for nsml
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--iteration", type=str, default='0')
    parser.add_argument("--pause", type=int, default=0)

    parser.add_argument('--dry_run', type=bool, default=False)

    config = parser.parse_args()
    main(config)
