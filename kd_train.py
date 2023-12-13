import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
# from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.optim import lr_scheduler, optimizer
from pytorch_metric_learning import distances, losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LambdaLR
import utils

import numpy as np
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from dataloaders.GSVCitiesDataset import GSVCitiesDataset
from dataloaders import MapillaryDataset
from dataloaders import PittsburgDataset
from models import helper
import glob

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

VIT_MEAN_STD = {'mean': [0.5, 0.5, 0.5], 
                'std': [0.5, 0.5, 0.5]}

TRAIN_CITIES = [
    'Bangkok',
    'BuenosAires',
    'LosAngeles',
    'MexicoCity',
    'OSL',
    'Rome',
    'Barcelona',
    'Chicago',
    'Madrid',
    'Miami',
    'Phoenix',
    'TRT',
    'Boston',
    'Lisbon',
    'Medellin',
    'Minneapolis',
    'PRG',
    'WashingtonDC',
    'Brussels',
    'London',
    'Melbourne',
    'Osaka',
    'PRS',
]

class VPRModel(nn.Module):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                #---- Backbone
                backbone_arch='resnet50',
                pretrained=True,
                layers_to_freeze=1,
                layers_to_crop=[],
                
                #---- Aggregator
                agg_arch='ConvAP', #CosPlace, NetVLAD, GeM
                agg_config={},
                
                #---- Train hyperparameters
                lr=0.03, 
                optimizer='sgd',
                weight_decay=1e-3,
                momentum=0.9,
                warmpup_steps=500,
                milestones=[5, 10, 15],
                lr_mult=0.3,
                
                #----- Loss
                loss_name='MultiSimilarityLoss', 
                miner_name='MultiSimilarityMiner', 
                miner_margin=0.1,
                faiss_gpu=False
                 ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop

        self.agg_arch = agg_arch
        self.agg_config = agg_config

        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        # self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)
        
    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x
    
    
def load_model(ckpt_path = 'resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt'):
        # Note that images must be resized to 320x320
        model = VPRModel(backbone_arch='resnet50',
                        layers_to_crop=[4],
                        agg_arch='MixVPR',
                        agg_config={'in_channels': 1024,
                                    'in_h': 20,
                                    'in_w': 20,
                                    'out_channels': 1024,
                                    'mix_depth': 4,
                                    'mlp_ratio': 1,
                                    'out_rows': 4},
                        )

        state_dict = torch.load(ckpt_path)
        model.load_state_dict(state_dict)

        model.eval()
        print(f"Loaded model from {ckpt_path} Successfully!")
        return model

def loss_function(descriptors, labels, miner, loss_fn):
        # we mine the pairs/triplets if there is an online mining strategy
        if miner is not None:
            miner_outputs = miner(descriptors, labels)
            loss = loss_fn(descriptors, labels, miner_outputs)
            
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            loss = loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner objet), 
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        # batch_accs.append(batch_acc)
        # log it
        # log('b_acc', sum(batch_acc) /
        #         len(batch_acc), prog_bar=True, logger=True)
        return loss

def train(model, model_dis, train_loader, optimizer, scheduler, miner, loss_fn, device):
    print('started training')
    running_loss = 0.0
    iter_cnt = 0
    correct_sum = 0
    
    model.train()
    model_dis.eval()
    soft_max = nn.Softmax(dim=1)
    total_loss = []
    img_path = '../datasets/gsv_cities/Images'
    img_path_list = glob.glob(img_path + '/**/**/*.jpg', recursive=True)
    for batch_i, batch in enumerate(train_loader):
        # print(f'batch index: {batch_i}')
        places, labels = batch
        
        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape
        
        # reshape places and labels
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)

        images = images.to(device)
        labels = labels.to(device)

        # Feed forward the batch to the model
        descriptors = model(images) # Here we are calling the method forward that we defined above
        
        # forward teacher model
        with torch.no_grad():
            global_descriptors = np.zeros((len(img_path_list), 4096))
            # for batch in tqdm(train_loader, ncols=100, desc=f'Extracting features'):
                # imgs, indices = batch
                # imgs = imgs.to(device)

                # model inference
            descriptors_dis = model_dis(images)
            # descriptors_dis = descriptors_dis.detach().cpu().numpy()

            # add to global descriptors
            # global_descriptors[np.array(indices), :] = descriptors

        
        loss_student =loss_function(descriptors, labels, miner, loss_fn)
        loss_teacher = loss_function(descriptors_dis, labels, miner, loss_fn)

        loss = loss_student+loss_teacher

        running_loss +=loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('finished training this epoch')
    scheduler.step()
    running_loss = running_loss / iter_cnt
    return running_loss

def test(model, test_loader, val_dataset, val_set_name, device):
    print('validating started')
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0
        data_num = 0

        validation_step=[]
        for batch_i, batch in enumerate(test_loader):
            places, _ = batch
            places = places.to(device)
            # calculate descriptors
            descriptors = model(places)
            descriptors = descriptors.detach().cpu()
            validation_step.append(descriptors)

        feats = torch.concat(validation_step, dim=0)
        if 'pitts' in val_set_name:
            # split to ref and queries
            num_references = val_dataset.dbStruct.numDb
            num_queries = len(val_dataset)-num_references
            positives = val_dataset.getPositives()
        elif 'msls' in val_set_name:
            # split to ref and queries
            num_references = val_dataset.num_references
            num_queries = len(val_dataset)-num_references
            positives = val_dataset.pIdx

        r_list = feats[ : num_references]
        q_list = feats[num_references : ]
        pitts_dict = utils.get_validation_recalls(r_list=r_list, 
                                            q_list=q_list,
                                            k_values=[1, 5, 10, 15, 20, 50, 100],
                                            gt=positives,
                                            print_results=True,
                                            dataset_name=val_set_name,
                                            faiss_gpu=False
                                            )
        del r_list, q_list, feats, num_references, positives
        
        print(f'R@1: {pitts_dict[1]}     | R@5: {pitts_dict[5]}       |R@10: {pitts_dict[10]}       ')
        # running_loss = running_loss / iter_cnt
    return pitts_dict

if __name__ == '__main__':
    
    epoch=80
    device = torch.device('cuda:1')
    # datamodule = GSVCitiesDataModule(
    #     batch_size=64,
    #     img_per_place=4,
    #     min_img_per_place=4,
    #     shuffle_all=False, # shuffle all images or keep shuffling in-city only
    #     random_sample_from_each_place=True,
    #     image_size=(320, 320),
    #     num_workers=28,
    #     show_data_stats=True,
    #     val_set_names=['msls_val'], # pitts30k_val, pitts30k_test, msls_val
    # )
    image_size = (320, 320)
    batch_size = 128
    num_workers = 4
    train_loader_config = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False}
    train_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN_STD['mean'], std=IMAGENET_MEAN_STD['std']),
        ])
    train_dataset = GSVCitiesDataset(
            cities=TRAIN_CITIES,
            img_per_place=4,
            min_img_per_place=4,
            random_sample_from_each_place=True,
            transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset, **train_loader_config)

    valid_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN_STD['mean'], std=IMAGENET_MEAN_STD['std'])])
    valid_loader_config = {
            'batch_size': batch_size,
            'num_workers': num_workers//2,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False}
    val_dataset = MapillaryDataset.MSLS(input_transform=valid_transform)
    val_loader = DataLoader(dataset=val_dataset, **valid_loader_config)
    val_dataset2 = PittsburgDataset.get_whole_val_set(input_transform=valid_transform)
    val_loader2 = DataLoader(dataset=val_dataset2, **valid_loader_config)
    val_dataset3 = PittsburgDataset.get_whole_test_set(input_transform=valid_transform)
    val_loader3 = DataLoader(dataset=val_dataset3, **valid_loader_config)
    # load model
    model_path = '/data/VCL/Nadhira/class/MixVPR/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt'
    # model_path = '/data/VCL/Nadhira/class/MixVPR/resnet50_MixVPR_128_channels(64)_rows(2).ckpt'
    model_teacher = load_model(model_path)
    
    model_student = VPRModel(
        #---- Encoder
        backbone_arch='resnet18',
        pretrained=True,
        layers_to_freeze=2,
        layers_to_crop=[4], # 4 crops the last resnet layer, 3 crops the 3rd, ...etc

        agg_arch='MixVPR',
        agg_config={'in_channels' : 256,
                'in_h' : 20,
                'in_w' : 20,
                'out_channels' : 128,
                'mix_depth' : 4,
                'mlp_ratio' : 1,
                'out_rows' : 4},  # the output dim will be (out_rows * out_channels)
                )

    lr=0.05 # 0.0002 for adam, 0.05 or sgd (needs to change according to batch size)
    
    weight_decay=0.001 # 0.001 for sgd and 0 for adam,
    momentum=0.9
    warmpup_steps=650
    milestones=[5, 10, 15, 25, 45]
    lr_mult=0.3
    optimizer = torch.optim.SGD(model_student.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=lr_mult)
    #----- Loss functions
    # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
    # FastAPLoss, CircleLoss, SupConLoss,
    loss_name='MultiSimilarityLoss'
    miner_name='MultiSimilarityMiner' # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
    miner_margin=0.1
    faiss_gpu=False

    loss_fn = utils.get_loss(loss_name)
    miner = utils.get_miner(miner_name, miner_margin)
    
    model_student.to(device)
    model_teacher.to(device)
    
    for i in range(0, epoch):
        print(f'Epoch: {i}')
        train_loss = train(model_student, model_teacher, train_loader, optimizer, scheduler, miner, loss_fn, device)
        val_result1 = test(model_student, val_loader2, val_dataset2, ['pitts'], device)
        val_result2 = test(model_student, val_loader3, val_dataset3, ['pitts'], device)
        val_result3 = test(model_student, val_loader, val_dataset, ['msls'], device)
        file_path = f'kd_model_pitts/kd_epoch{i}.ckpt'
        torch.save(model_student.state_dict(), file_path)
    

