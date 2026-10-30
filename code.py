import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import StdConv2dSame, DropPath, to_2tuple, trunc_normal_
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models # add models to the list
from torchvision.utils import make_grid
#from torchvision.models import vision_transformer
import matplotlib.pyplot as plt
import math
import time
import os
import timm
from datetime import timedelta
from sklearn.metrics import confusion_matrix
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
test_acc = []
val_corrects = []
train_corrects = []
train_loss = []
val_loss = []
start = 7
end = 8
epochs = 50
lr = 1.651566109751641 * (10 ** (-5))
drp = 0.0809990187955752
ptch_sz = 16

space= {
    'learning_rate': hp.loguniform('learning_rate', np.log(10 ** (-6)), np.log(10 ** (-4))),
    'dropout' : hp.uniform('dropout', 0, 0.1)
}

lr_pos = [10 ** (-8), 10 ** (-7), 10 ** (-6), 10 ** (-5), 10 ** (-4), 10 ** (-3)]
drop_pos = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
patch_pos = [16]
class patchembed(nn.Module):
    def __init__(self, img_size, patch_size, chans, embed_dim):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[0] // patch_size[0]) ** 2
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.c1 = nn.Conv2d(chans, embed_dim, patch_size, patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        return self.c1(x).flatten(2).transpose(1, 2)

class self_attention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias = False, qk_scale = None, attn_drop = 0, proj_drop = 0):
        super().__init__()
        self.num_heads=  num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias = qkv_bias)
        self.drop1 = nn.Dropout(attn_drop)
        self.mlp = nn.Linear(embed_dim, embed_dim)
        self.drop2 = nn.Dropout(proj_drop)

    def forward(self, x):
        btch, patch_num, patch_dim = x.shape
        qkv = self.qkv(x).reshape(btch, patch_num, 3, self.num_heads, patch_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim = -1)
        attn = self.drop1(attn)
        res = (attn @ v).transpose(1, 2).reshape(btch, patch_num, patch_dim)
        res = self.mlp(res)
        res = self.drop2(res)
        return res
class ffn(nn.Module):
    def __init__(self, dim, hidden_features = None, dropout = 0):
        super().__init__()
        if(hidden_features == None):
            hidden_features = dim
        self.fc1 = nn.Linear(dim, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio = 4, qkv_bias = False, qk_scale = None, drop = 0, attn_drop = 0, drop_path = 0, norm_layer = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = self_attention(dim, num_heads, qkv_bias = qkv_bias, qk_scale = qk_scale, attn_drop = attn_drop, proj_drop = drop)
        self.norm2 = norm_layer(dim)
        self.mlp = ffn(dim, hidden_features = dim * mlp_ratio)
        self.drop_path = nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class ViT(nn.Module):
    def __init__(self, embed_dim = 768, img_size = 224, patch_size = 16, in_chans = 3, num_classes = 1000, num_blocks = 12, num_heads = 12, mlp_ratio = 4, qkv_bias = None,
               qk_scale = None, drop = 0., attn_drop = 0., drop_path = 0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = patchembed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # first 1 is the batch size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop)

        depth_decay = [x.item() for x in torch.linspace(0, drop_path, num_blocks)]

        self.encoders = nn.Sequential(
            *[block(dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias = qkv_bias, qk_scale = qk_scale,
                      drop = drop, attn_drop = attn_drop, drop_path = depth_decay[i], norm_layer = nn.LayerNorm) for i in range(num_blocks)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.pre_logits = nn.Identity()

        self.mlphead = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim = 1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.encoders(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pre_logits(x[:, 0])
        x = self.mlphead(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, 3, 1)
        self.conv2 = nn.Conv2d(24, 120, 3, 1)
        self.fc1 = nn.Linear(54*54*120, 120)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54*54*120)
        X = self.dropout(X)
        X = self.dropout(F.relu(self.fc1(X)))
        X = self.dropout(F.relu(self.fc2(X)))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)
train_transform = transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
        #transforms.Normalize([0.485, 0.456, 0.406],
                             #[0.229, 0.224, 0.225])
    ])
def get_data(filename, batch_size): 
    torch.manual_seed(101)
    root = filename
    global train_data, val_data, test_data, train_loader, val_loader, test_loader
    train_data = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_transform)
    train_data2 = datasets.ImageFolder(os.path.join(root, 'train'), transform=transforms.ToTensor())
    val_data = datasets.ImageFolder(os.path.join(root, 'valid'), transform=test_transform)
    test_data = datasets.ImageFolder(os.path.join(root, 'test'), transform=test_transform)

    torch.manual_seed(42)
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = len(test_data), shuffle=True)
    return train_data, val_data, test_data, train_loader, val_loader, test_loader
def check_dataset(): 
    class_names = train_data.classes
    print(class_names)
    print(f'Training images available: {len(train_data)}')
    print(f'Val images available:  {len(val_data)}')
    print(f'Testing images available:  {len(test_data)}')
    for images,labels in train_loader:
        break

    print('Label:', labels.numpy())
    print('Class:', *np.array([class_names[i] for i in labels]))

    im = make_grid(images, nrow=5)

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    im_inv = inv_normalize(im)

    plt.figure(figsize=(12,4))
    plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)));
def train_model(learning_rate, epochs):
    
    #model = ViT(num_blocks = 8, num_heads = 8, mlp_ratio = 2, drop = 0.4, attn_drop = 0.4, drop_path = 0.4)
    model = ViT(num_classes = 2, drop = drp, drop_path=drp, patch_size = ptch_sz)
    #model = ViT(num_classes = 2, weights="IMAGENET1K_V2")
    #model = vision_transformer.vit_base_patch16_224(pretrained=True)
    #model = timm.create_model('vit_base_patch16_224', pretrained=True)
    #model = CNN()    
    #for param in model.parameters():
     #   param.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    #optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)
    min_r = 0.5
    lr_current = 0
    s_func = lambda epoch: 0.5 * (1+math.cos(epoch*math.pi/epochs)) * (1-min_r) + min_r
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=s_func)
    start_time = time.time()

    train_losses = []
    val_losses = []
    train_correct = []
    val_correct = []
    global predicted
    for epoch in range(epochs):
        trn_corr = 0
        val_corr = 0
        loss_sum = 0
        model.train()
        # b is the batch number
        for b, (X_train, y_train) in enumerate(train_loader):
            b+=1
            optimizer.zero_grad()
            y_pred = model.forward(X_train)
            loss = criterion(y_pred, y_train)
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr

            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        loss_sum /= b
        train_losses.append(loss_sum)
        train_correct.append(trn_corr)
        print(f'epoch: {epoch:2} loss: {loss_sum:10.8f}  \
  accuracy: {trn_corr.item()*100/(len(train_data)):7.3f}%')
        loss_sum = 0
        # Run the validation batches
        with torch.no_grad():
            model.eval()
            for b, (X_val, y_val) in enumerate(val_loader):
                b += 1
                y_ans = model.forward(X_val)
                predicted = torch.max(y_ans.data, 1)[1]
                val_corr += (predicted == y_val).sum()
                loss = criterion(y_ans, y_val)
                loss_sum += loss

        val_losses.append(loss_sum / b)
        print(loss_sum / b)
        val_correct.append(val_corr)
        if scheduler:
            scheduler.step()                # Change the learning rate by time
            lr_current = scheduler.get_last_lr()[0]
        else:
            lr_current = optimizer.state_dict()['param_groups'][0]['lr']
    total_time = time.time() - start_time
    #Run on the test set
    with torch.no_grad():
        model.eval()
        tst_corr = 0
        global y_test
        for b, (X_test, y_test) in enumerate(test_loader):

         # Apply the model
            y_pre = model.forward(X_test)

        # Tally the number of correct predictions
            predicted = torch.max(y_pre.data, 1)[1]
            tst_corr += (predicted == y_test).sum()

        loss = criterion(y_pre, y_test)
    print(tst_corr / len(test_data) * 100)
    return train_losses, train_correct, val_losses, val_correct, tst_corr / len(test_data) * 100, total_time
def train_ViT_model():
    #t_l ---> train losses, t_c --> train corrects
    #v_l --> validation losses, v_c --> validation corrects
    t_l, t_c, v_l, v_c, t_acc, time = train_model(lr, epochs)
    train_loss.append(t_l)
    train_corrects.append(t_c)
    val_loss.append(v_l)
    val_corrects.append(v_c)
    test_acc.append(t_acc)
    times = str(timedelta(seconds=time)).split(':')
    print(f'\nDuration: {times[0]} hours, {times[1]} minutes, {times[2]} seconds') # print the time elapsed
    print(t_acc)

def save_results():
    ls = []
    col = ['Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy']

    for i in range(epochs):
        tmp = []
        tmp.append(round(train_loss[0][i], 3))
        tmp.append(round(train_corrects[0][i].item(), 3))
        tmp.append(round(val_loss[0][i].item(), 3))
        tmp.append(round(val_corrects[0][i].item(), 3))
        ls.append(tmp)   

    df_ls = pd.DataFrame(ls, columns = col)
    df_ls.head()
    df_ls.to_csv("vit_bayesian.csv")
def plot_training_loss():
    plt.subplot(221)
    plt.plot(train_loss[0])
    plt.xlabel('Training Loss at the end of each epoch')
    plt.legend()
def plot_val_loss():
    plt.subplot(222)
    plt.plot(val_loss[0])
    plt.xlabel('Validation Loss at the end of each epoch')
    plt.legend()
def plot_train_acc():
    plt.subplot(223)
    train_acc = [j/len(train_data)*100 for j in train_corrects[0]]
    plt.plot(train_acc)
    plt.xlabel('Training Accuracy')
    plt.legend()
def plot_val_acc():
    plt.subplot(224)
    val_acc = [j/len(val_data)*100 for j in val_corrects[0]]
    plt.plot(val_acc)
    plt.xlabel('Validation Accuracy')
    plt.legend()
def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    print(f'______\n{sum(params):>6}')
def bayesian(ps):
    lr, drp_tmp = ps['learning_rate'], ps['dropout']
    max_trn_batch = 10
    model = ViT(num_classes = 2, drop = drp_tmp, drop_path = drp_tmp)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    min_r = 0.5
    s_func = lambda epoch: 0.5 * (1+math.cos(epoch*math.pi/epochs)) * (1-min_r) + min_r
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=s_func)
    for epoch in range(epochs):
        model.train()
        # b is the batch number
        for b, (X_train, y_train) in enumerate(train_loader):
            b+=1
            if(b == max_trn_batch):
                break
            optimizer.zero_grad()
            y_pred = model.forward(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
        scheduler.step()
    model.eval()
    with torch.no_grad():
        loss_sum = 0
        for b, (X_val, y_val) in enumerate(val_loader):
            b += 1
            optimizer.zero_grad()
            y_pred = model.forward(X_val)
            loss = criterion(y_pred, y_val)
            loss_sum += loss
    return float(loss_sum / b)
    
def random_search():
    lr = 10 ** (np.random.uniform(-8, -3))
    ep = np.random.randint(3, 31)
    max_trn_batch = 10
    model = ViT(num_classes = 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    min_r = 0.5
    s_func = lambda epoch: 0.5 * (1+math.cos(epoch*math.pi/epochs)) * (1-min_r) + min_r
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=s_func)
    for epoch in range(ep):
        model.train()
        # b is the batch number
        for b, (X_train, y_train) in enumerate(train_loader):
            b+=1
            if(b == max_trn_batch):
                break
            optimizer.zero_grad()
            y_pred = model.forward(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
        scheduler.step()
    model.eval()
    with torch.no_grad():
        loss_sum = 0
        for b, (X_val, y_val) in enumerate(val_loader):
            b += 1
            optimizer.zero_grad()
            y_pred = model.forward(X_val)
            loss = criterion(y_pred, y_val)
            loss_sum += loss
    return [float(loss_sum / b), lr, ep]
def grid_search(lr_tmp, drop_tmp, patch_tmp):
    max_trn_batch = 10
    model = ViT(num_classes = 2, drop = drop_tmp, attn_drop = 0, drop_path = drop_tmp, patch_size=patch_tmp)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_tmp, betas=(0.9, 0.999))
    min_r = 0.5
    s_func = lambda epoch: 0.5 * (1+math.cos(epoch*math.pi/epochs)) * (1-min_r) + min_r
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=s_func)
    for epoch in range(epochs):
        model.train()
        # b is the batch number
        for b, (X_train, y_train) in enumerate(train_loader):
            b+=1
            if(b == max_trn_batch):
                break
            optimizer.zero_grad()
            y_pred = model.forward(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
        scheduler.step()
    model.eval()
    with torch.no_grad():
        loss_sum = 0
        for b, (X_val, y_val) in enumerate(val_loader):
            b += 1
            optimizer.zero_grad()
            y_pred = model.forward(X_val)
            loss = criterion(y_pred, y_val)
            loss_sum += loss
    return float(loss_sum / b)
start_time = time.time()
tr_d, v_d, ts_d, tr_l, v_l, ts_l = get_data('vit2.v1i.folder', 10)
#tr_d, v_d, ts_d, tr_l, v_l, ts_l = get_data('data_1', 3)
check_dataset()

num_eval = 50
trials = Trials()
best_param = fmin(bayesian, space = space, algo=tpe.suggest, max_evals=num_eval, trials=trials)
lr, drp = best_param['learning_rate'], best_param['dropout']
print(lr, drp)

epochs = 100
plt.figure(figsize = (12, 8))
train_ViT_model()
#save_loss()
#save_acc()
save_results()
plot_training_loss()
plot_val_loss()
plot_train_acc()
plot_val_acc()
tn, fp, fn, tp = confusion_matrix(y_test.view(-1), predicted.view(-1)).ravel()
print(tn, fp, fn, tp)
f_1 = (2 * tp) / (2 * tp + fp + fn)
print("F1 score:", f_1)
end_time = time.time()
end_time = end_time - start_time
times = str(timedelta(seconds=end_time)).split(':')
print(f'\nDuration: {times[0]} hours, {times[1]} minutes, {times[2]} seconds')
print(confusion_matrix(y_test.view(-1), predicted.view(-1)))
tn, fp, fn, tp = confusion_matrix(y_test.view(-1), predicted.view(-1)).ravel()
print(tn, fp, fn, tp)
def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    print(f'______\n{sum(params):>6}')
model = ViT(num_classes = 2)
count_parameters(model)
save_results()
plot_training_loss()
plot_val_loss()
plot_train_acc()
plot_val_acc()
tn, fp, fn, tp = confusion_matrix(y_test.view(-1), predicted.view(-1)).ravel()
print(tn, fp, fn, tp)
f_1 = (2 * tp) / (2 * tp + fp + fn)
print("F1 score:", f_1)
end_time = time.time()
end_time = end_time - start_time
times = str(timedelta(seconds=end_time)).split(':')
print(f'\nDuration: {times[0]} hours, {times[1]} minutes, {times[2]} seconds')
