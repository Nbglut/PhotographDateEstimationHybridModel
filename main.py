from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import json
import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler
import os
from torch.utils.data import DataLoader
import wandb
from utils import str2bool
from torchmetrics import Precision
from torchmetrics import Recall
import argparse
from collections import Counter
from collections import OrderedDict
from torchvision import datasets
from PIL import Image, ImageFile
import time
from torchvision import models
from torchvision.models.vision_transformer import EncoderBlock
from CNN import ConvEncoder
from model import CNNModel
import timm



#API KEY WANDB ##42c65da3d76f8de5a236bb06dab102275096401b

## input hyper-paras
## input hyper-paras
parser = argparse.ArgumentParser(description = "nueral networks")
parser.add_argument("-mode", dest="mode", type=str, default='train', help="train or test")
parser.add_argument("-num_epoches", dest="num_epoches", type=int, default=30, help="num of epoches")
parser.add_argument("-learning_rate", dest ="learning_rate", type=float, default=0.0001, help = "learning rate")
parser.add_argument("-batch_size", dest="batch_size", type=int, default=16, help="batch size")
parser.add_argument("-load_checkpoint", dest="load_checkpoint", type=str2bool, default=False, help="true of false")
parser.add_argument("-model", dest="model", type=str, default='basic', help="basic or CNN")




parser.add_argument("-ckp_path", dest='ckp_path', type=str, default="checkpoint", help="path of checkpoint")

Image.MAX_IMAGE_PIXELS = None  # Remove pixel limit entirely
ImageFile.LOAD_TRUNCATED_IMAGES = True

args = parser.parse_args()


def _load_data(batch_size):
    '''Data loader'''
#Transformations
    print("start loading....")
    transformations = transforms.Compose([
    transforms.Pad((10, 10, 10, 10)),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.RandomAffine(degrees=50, shear=(-20, 20), translate=(0.2, 0.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.3),
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#load datasets
    print("Load training")
    dataset= datasets.ImageFolder(root='data/train',transform=transformations)
    print("Load test")
    testdataset= datasets.ImageFolder(root='data/test',transform=transformations)
    print("Done!")
#Oversampling / Undersampling
    class_num = Counter(dataset.targets)
    print(f"Class counts: {class_num}")
    total_sam = sum(class_num.values())
    print(f"total sam: {total_sam}")

    cls_weights = {cls: total_sam / count for cls, count in class_num.items()}
    max_weight = max(cls_weights.values())
    cls_weights = {cls: weight / max_weight for cls, weight in cls_weights.items()}


    print(f"class weights: {cls_weights}")

    samp_weights = [cls_weights[label] for label in dataset.targets] 
    print(f"Samp weights calculated.")
    samps = WeightedRandomSampler(weights=samp_weights, num_samples=len(dataset), replacement=True)

    print("Load training.")
    train_loader= torch.utils.data.DataLoader(dataset, batch_size=32, sampler=samps)
    print("Load test.")
    test_loader=torch.utils.data.DataLoader(testdataset,batch_size=32, shuffle=True)
    print("Done loading.")
    return train_loader, test_loader


def _compute_accuracy(y_pred, y_batch):
    accy = (100 * torch.eq(y_batch, y_pred).sum().item()) // y_batch.size(0)
    return accy


    


def adjust_learning_rate(learning_rate, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 1/10 every args.lr epochs"""
    lr = learning_rate
  
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # print("learning_rate: ", lr)
    
    
def main():

    use_cuda = torch.cuda.is_available() ## if have gpu or cpu 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: ", device)
    if use_cuda:
        torch.cuda.manual_seed(72)
    wandbname="PhotographicEstimationViT"
    ## initialize hyper-parameters
    num_epoches = args.num_epoches
    learning_rate = args.learning_rate
    
    train_loader, test_loader=_load_data( args.batch_size)
    
    #if model is ViT base 
    model= models.vit_b_16(pretrained=True)  # ViT-Base 
    heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
   #Make VIT output 13 classes by changing last layer
    heads_layers["pre_logits"] = nn.Linear(768 , 512)
    heads_layers["act"] = nn.Tanh()
    heads_layers["head"] = nn.Linear(512, 13)
    model.heads = nn.Sequential(heads_layers)
#else if model is CNN 
    if args.model=='CNNViT':
#change the Encoder by making a "CNNBlock" which is the same as normal TransformerBlock but with  CNNs instead of MLPs
         model.encoder=  ConvEncoder(
            model.seq_length,
            16, #num_layers
            12, #num_heads
            model.hidden_dim,
            model.mlp_dim,
            model.dropout,
            model.attention_dropout,
            model.norm_layer,
        )
         wandbname=CNNVit
    if args.model=='CNN':
#CNN        wandbname="CNN" 
            #CNN
            model= torchvision.models.vgg16(pretrained=True)
            #Change to 13 classes
            num_ftrs = model.classifier[6].in_features  
            model.classifier[6] = torch.nn.Linear(num_ftrs, 13) 

            print("BASIC CNN MODEL") 
    if args.model=="CNNX":
           wandbname='CNNX'
           print("CNNX!")
           model = timm.create_model('xception',num_classes=13, pretrained=True) 






    ## to gpu or cpu
    model.to(device)
    wandb.init(project='PhotographicEstimation', name=wandbname)
    wandb.watch(model, log_freq=100)



    optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-4)  ## Adam Optimizer
#Weighted loss function
    class_num = Counter(train_loader.dataset.targets)
    total_sam = sum(class_num.values())

    cls_weights = {cls: total_sam / count for cls, count in class_num.items()}
    max_weight = max(cls_weights.values())
    cls_weights = {cls: weight / max_weight for cls, weight in cls_weights.items()}
    class_weights = torch.tensor([cls_weights[i] for i in range(13)]).to(device)

    loss_fun =   nn.CrossEntropyLoss(weight=class_weights)  ## cross-entropy loss
    #loss_fun =   nn.CrossEntropyLoss()  ## cross-entropy loss

    ## load checkpoint 
    curr_dir=os.path.dirname(__file__)
    filename= 'last.pt' 
    path= os.path.join(curr_dir, filename)
    if args.load_checkpoint == True:
        ## write load checkpoint code below
        checkpoint=torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoches=checkpoint['epoch'] 
        print('Checkpoint loaded!')
    
    ##  model training
    if args.mode == 'train':
        print("Start training")
        model = model.train()
        for epoch in range(num_epoches): #10-50
            ## learning rate
            adjust_learning_rate(learning_rate, optimizer, epoch)

            for batch_id, (x_batch,y_labels) in enumerate(train_loader):
                x_batch,y_labels = Variable(x_batch).to(device), Variable(y_labels).to(device)
                
                ## feed input data x into model
                output_y = model(x_batch)
                
                #loss function
                loss =   loss_fun(output_y, y_labels ) 
                #back prop
                optimizer.zero_grad()
                loss.backward()
#Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                optimizer.step()

                #compute accuracy

                _, y_pred = torch.max(output_y.data, 1)
                accy = _compute_accuracy(y_pred, y_labels)
            
            
                #wandb
                wandb.log({'epoch':epoch, 'training accuracy' :accy})
                wandb.log({"training loss": loss})


                step =step = epoch * len(train_loader) + batch_id  # Track global step

            checkpoint={'epoch': epoch, 
             'global_setp':step,
             'model_state_dict' :model.state_dict(),
             'optimizer_state_dict':optimizer.state_dict()}
            torch.save(checkpoint, path)
            print(f'Checkpoint Saved at epoch {epoch}')



                

    print('Training Complete!')
    #test model
    model.eval()
    with torch.no_grad():
        for batch_id, (x_batch,y_labels) in enumerate(test_loader):
            x_batch, y_labels = Variable(x_batch).to(device), Variable(y_labels).to(device)
        
            output_y = model(x_batch)
            _, y_pred = torch.max(output_y.data, 1)

    
            accy = _compute_accuracy(y_pred, y_labels)
            precisioncalc = Precision(task='multiclass', average='macro', num_classes=13)
            precision = precisioncalc(y_pred, y_labels)
            recallcalc= Recall(task="multiclass", average='macro', num_classes=13)
            recall = recallcalc(y_pred, y_labels)

            f1 = 2*((precision * recall)/(precision + recall))

            wandb.log({'Evalulated accuracy' :accy})
            wandb.log({'F1 Score' :f1})
            wandb.log({'Recall' :recall})
            wandb.log({'Precision' :precision})


    

if __name__ == '__main__':
    main()
    



