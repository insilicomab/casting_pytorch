# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch import nn, optim

'''
定数の指定
'''

# 学習済みパラメータのパス
model_path = '/content/drive/My Drive/Colab Notebooks/SIGNATE/casting_pytorch/model/casting_pytorch_Epoch17_logloss_0.0002.pth'

# 提出データの保存先
submit_path = '/content/drive/My Drive/Colab Notebooks/SIGNATE/casting_pytorch/submit/casting_pytorch_mobilenet_v3_ver1.csv'

# 学習データのラベルマスター
test_labels = './sample_submission.csv'
sample_submit = './sample_submission.csv'

# 画像データのディレクトリ
img_dir = './test_data/'

# リサイズする画像サイズ
photo_size = 300

# クラス数の定義
num_classes = 2

# 学習に使用する機器(device)の設定
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'デバイス：{device}')

'''
学習済みパラメータの読み込み
'''

model = models.mobilenet_v3_large(pretrained=False)
fc_in_features = model.classifier[3].in_features # 最終レイヤー関数の次元数
model.fc = nn.Linear(fc_in_features, num_classes) # 最終レイヤー関数の付け替え

# モデルをGPUに送る
model.to(device)

# 学習済みパラメータの読み込み
trained_params = torch.load(model_path)

# モデルにパラメータをロード
model.load_state_dict(trained_params)

'''
データの読み込み
'''

# ラベルデータの読み込み
test_labels = pd.read_csv(test_labels, header=None, sep=',')
print(test_labels.head())

# 画像データの名前リストの抽出
x_test = test_labels[0].values
dummy = test_labels[0].values

'''
前処理とデータセットの作成
'''

# transformの設定
transform = {
    'train': transforms.Compose([
        transforms.Resize(photo_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    'val': transforms.Compose([
        transforms.Resize(photo_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
}

# Datasetの設定
class CastingDataset(Dataset):
    def __init__(self, image_name_list, label_list, img_dir, phase=None):
        self.image_name_list = image_name_list # 画像ファイル名
        self.label_list = label_list # ラベル
        self.img_dir = img_dir # 画像データのディレクトリ
        self.phase = phase # 変数phaseで学習(train)もしくは検証(val)の設定を行う
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list) # 1エポックあたりに読み込むデータ数として、入力データの数を指定

    def __getitem__(self, index):
        
        # index番目の画像を読み込み、前処理を行う
        image_path = os.path.join(self.img_dir, self.image_name_list[index]) # train_master.iloc[index, 0]はファイル名を抽出
        img = Image.open(image_path)
        img = self.transform[self.phase](img)
        
        # index番目のラベルを取得する
        label = self.label_list[index]
        
        return img, label

# Datasetのインスタンス作成
test_dataset = CastingDataset(x_test, dummy, img_dir, phase='val')

# Dataloader
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

'''
テストデータの予測
'''

# 予測データフレームの作成
preds = []

# dataloaderから、ミニバッチ単位でデータを読み込む
for images, _ in test_dataloader:
    
    # 入力データをdeviceへ
    images = images.to(device)
    
    # 学習済みモデルを推論モードに設定
    model.eval()
    
    # モデルによる変換
    outputs = model(images)
    pred = torch.argmax(outputs, dim=1)
    pred = pred.to('cpu').numpy()

    # 予測値をリストに追加
    preds.extend(pred)

'''
提出
'''

# 提出用データの読み込み
sub = pd.read_csv(sample_submit, header=None, sep=',')
print(sub.head())

# 目的変数カラムの置き換え
sub[1] = preds

# ファイルのエクスポート
sub.to_csv(submit_path, sep=',', header=None, index=None)