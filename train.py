import torch.nn.functional as F
import torchvision.transforms as T
from pytorch_metric_learning import distances, losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LambdaLR
from visual_places import get_visual_place, MixVPRVisualPlace

import numpy as np
from models import aggregators
from models import backbones
from evaluate import eval_fn
from visual_places import VisualPlaceImage
import wandb

from datetime import datetime
def get_backbone(
    backbone_arch="resnet50",
    pretrained=True,
    layers_to_freeze=2,
    layers_to_crop=[],
):
    """Helper function that returns the backbone given its name

    Args:
        backbone_arch (str, optional): . Defaults to 'resnet50'.
        pretrained (bool, optional): . Defaults to True.
        layers_to_freeze (int, optional): . Defaults to 2.
        layers_to_crop (list, optional): This is mostly used with ResNet where
                                         we sometimes need to crop the last
                                         residual block (ex. [4]). Defaults to [].

    Returns:
        nn.Module: the backbone as a nn.Model object
    """
    if "resnet" in backbone_arch.lower():
        return backbones.ResNet(
            backbone_arch, pretrained, layers_to_freeze, layers_to_crop
        )

    elif "efficient" in backbone_arch.lower():
        if "_b" in backbone_arch.lower():
            return backbones.EfficientNet(
                backbone_arch, pretrained, layers_to_freeze + 2
            )
        else:
            return backbones.EfficientNet(
                model_name="efficientnet_b0",
                pretrained=pretrained,
                layers_to_freeze=layers_to_freeze,
            )

    elif "swin" in backbone_arch.lower():
        return backbones.Swin(
            model_name="swinv2_base_window12to16_192to256_22kft1k",
            pretrained=pretrained,
            layers_to_freeze=layers_to_freeze,
        )


def get_aggregator(agg_arch="ConvAP", agg_config={}):
    """Helper function that returns the aggregation layer given its name.
    If you happen to make your own aggregator, you might need to add a call
    to this helper function.

    Args:
        agg_arch (str, optional): the name of the aggregator. Defaults to 'ConvAP'.
        agg_config (dict, optional): this must contain all the arguments needed to instantiate the aggregator class. Defaults to {}.

    Returns:
        nn.Module: the aggregation layer
    """

    if "cosplace" in agg_arch.lower():
        assert "in_dim" in agg_config
        assert "out_dim" in agg_config
        return aggregators.CosPlace(**agg_config)

    elif "gem" in agg_arch.lower():
        if agg_config == {}:
            agg_config["p"] = 3
        else:
            assert "p" in agg_config
        return aggregators.GeMPool(**agg_config)

    elif "convap" in agg_arch.lower():
        assert "in_channels" in agg_config
        return aggregators.ConvAP(**agg_config)

    elif "mixvpr" in agg_arch.lower():
        assert "in_channels" in agg_config
        assert "out_channels" in agg_config
        assert "in_h" in agg_config
        assert "in_w" in agg_config
        assert "mix_depth" in agg_config
        return aggregators.MixVPR(**agg_config)


def get_loss(loss_name):
    if loss_name == "SupConLoss":
        return losses.SupConLoss(temperature=0.07)
    if loss_name == "CircleLoss":
        return losses.CircleLoss(
            m=0.4, gamma=80
        )  # these are params for image retrieval
    if loss_name == "MultiSimilarityLoss":
        return losses.MultiSimilarityLoss(
            alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity()
        )
    if loss_name == "ContrastiveLoss":
        return losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    if loss_name == "Lifted":
        return losses.GeneralizedLiftedStructureLoss(
            neg_margin=0, pos_margin=1, distance=DotProductSimilarity()
        )
    if loss_name == "FastAPLoss":
        return losses.FastAPLoss(num_bins=30)
    if loss_name == "NTXentLoss":
        return losses.NTXentLoss(
            temperature=0.07
        )  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
    if loss_name == "TripletMarginLoss":
        return losses.TripletMarginLoss(
            margin=0.1, swap=False, smooth_loss=False, triplets_per_anchor="all"
        )  # or an int, for example 100
    if loss_name == "CentroidTripletLoss":
        return losses.CentroidTripletLoss(
            margin=0.05,
            swap=False,
            smooth_loss=False,
            triplets_per_anchor="all",
        )
    raise NotImplementedError(f"Sorry, <{loss_name}> loss function is not implemented!")


class MixVPRModel(nn.Module):
    def __init__(self, backbone, aggregator):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator

    def forward(self, x):
        return self.aggregator(self.backbone(x))


def collate_fn(batch: List[dict]):
    images = [b["images"] for b in batch]
    ids = [b["id"] for b in batch]

    labels = torch.as_tensor(ids)
    images = torch.stack(images)
    return {"images": images, "labels": labels}


#  The loss function call (this method will be called at each training iteration)
def loss_fn(miner, metric_fn, descriptors, labels):
    # we mine the pairs/triplets if there is an online mining strategy
    miner_outputs = miner(descriptors, labels)
    loss = metric_fn(descriptors, labels, miner_outputs)

    # calculate the % of trivial pairs/triplets
    # which do not contribute in the loss value
    nb_samples = descriptors.shape[0]
    nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
    batch_acc = 1.0 - (nb_mined / nb_samples)

    return {"loss": loss, "batch_acc": batch_acc}


# This is the training step that's executed at each iteration
def training_step(model, loss_fn, batch):
    places, labels = batch

    # Note that GSVCities yields places (each containing N images)
    # which means the dataloader will return a batch containing BS places
    BS, N, ch, h, w = places.shape

    # reshape places and labels
    images = places.view(BS * N, ch, h, w)
    labels = labels.view(-1)

    # Feed forward the batch to the model
    descriptors = model(
        images
    )  # Here we are calling the method forward that we defined above
    outputs = loss_fn(descriptors, labels)  # Call the loss_function we defined above

    return outputs


def get_miner(miner_name, margin=0.1):
    if miner_name == "TripletMarginMiner":
        return miners.TripletMarginMiner(
            margin=margin, type_of_triplets="semihard"
        )  # all, hard, semihard, easy
    if miner_name == "MultiSimilarityMiner":
        return miners.MultiSimilarityMiner(
            epsilon=margin, distance=distances.CosineSimilarity()
        )
    if miner_name == "PairMarginMiner":
        return miners.PairMarginMiner(
            pos_margin=0.7, neg_margin=0.3, distance=distances.DotProductSimilarity()
        )
    return None


def expoential_lr(decay, start_lr, end_lr, warmup_steps, step):
    min_factor = end_lr / start_lr
    if step < warmup_steps:
        return step / warmup_steps
    else:
        return max(min_factor, decay ** (step - warmup_steps))


def train(
    dataset_name: str,
    root: str,
    epochs=100,
    backbone_arch="resnet50",
    pretrained=True,
    layers_to_freeze=1,
    layers_to_crop=[],
    # ---- Aggregator
    agg_arch="ConvAP",  # CosPlace, NetVLAD, GeM
    agg_config={},
    # ---- Train hyperparameters
    lr=0.03,
    optimizer="sgd",
    weight_decay=1e-3,
    momentum=0.9,
    warmpup_steps=500,
    milestones=[5, 10, 15],
    lr_mult=0.3,
    # ----- Loss
    loss_name="MultiSimilarityLoss",
    miner_name="MultiSimilarityMiner",
    miner_margin=0.1,
    faiss_gpu=False,
):
    hostname = os.uname()[1]
    wandb.init(
        project="VPR",
        name=f"MixVPR-{dataset_name}-{hostname}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        id=f"MixVPR-{dataset_name}-{hostname}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    image_size = 320
    metric_fn = get_loss(loss_name)
    miner = get_miner(miner_name, miner_margin)
    # batch_acc = (
    #    []
    # )  # we will keep track of the % of trivial pairs/triplets at the loss level
    faiss_gpu = faiss_gpu
    # ----------------------------------
    # get the backbone and the aggregator
    backbone = get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
    aggregator = get_aggregator(agg_arch, agg_config)
    model = MixVPRModel(backbone, aggregator)
    model.backbone.requires_grad_(False)
    model.load_state_dict(
        torch.load(
            "/scratch/zc2309/MixVPR/checkpoints/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt"
        )
    )
    model = model.to(memory_format=torch.channels_last).to("cuda")
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = LambdaLR(
        opt,
        lambda step: expoential_lr(lr_mult, lr, lr * lr_mult, warmpup_steps, step),
    )
    t = T.Compose(
        [
            T.Resize(image_size, antialias=True),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = get_visual_place(dataset_name, root, transform=t)
    dl = DataLoader(
        MixVPRVisualPlace.from_visual_place(dataset, transform=t),
        batch_size=8,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn,
    )
    step = 0
    os.makedirs(dataset_name, exist_ok=True)
    for epoch in range(epochs):
        # result = eval_fn(
        #    model,
        #    VisualPlaceImage.from_visual_place(dataset.sample(3000), transform=t),
        #    device="cuda",
        # )
        # print(result)
        # wandb.log(result, step=epoch)
        for i, batch in enumerate(pbar := tqdm(dl)):
            model.train()
            images = batch["images"].to("cuda")
            labels = batch["labels"].to("cuda")
            labels = labels.view(-1, 1).expand(-1, images.shape[1])
            images = images.flatten(0, 1).contiguous(memory_format=torch.channels_last)
            labels = labels.flatten(0, 1)
            with torch.autocast("cuda", torch.bfloat16):
                descriptors = model(images)
            opt.zero_grad()
            outputs = loss_fn(miner, metric_fn, descriptors, labels)
            outputs["loss"].backward()
            opt.step()
            sched.step()
            pbar.set_description(
                f"epoch {epoch} step {step} loss {outputs['loss'].item():.5f} batch_acc {outputs['batch_acc']:.5f} lr {sched.get_last_lr()[0]:.3e}"
            )
        model.eval()
        torch.save(model.state_dict(), f"{dataset_name}/model.pt")


if __name__ == "__main__":
    

    nyuvpr360 = {"dataset_name": "nyuvpr360", "root": "/datasets/run_0"}
    nordland = {
        "dataset_name": "nordland",
        "root": "/datasets/nordland_scenes/train",
    }
    msls = {"dataset_name": "msls", "root": "../datasets/msls_val/train_val"}

    kitti360 = {
        "dataset_name": "kitti360",
        "root": "/datasets/kitti360_scenes/2013_05_28_drive_0009_sync",
    }
    import sys

    # choice = sys.argv[1]
    #
    # data_config = None
    # if choice == "nyuvpr360":
    #    data_config = nyuvpr360
    # elif choice == "nordland":
    #    data_config = nordland
    # elif choice == "msls":
    #    data_config = msls
    # elif choice == "kitti360":
    #    data_config = kitti360
    config = dict(
        # ---- Encoder
        **msls,
        backbone_arch="resnet50",
        pretrained=True,
        layers_to_freeze=2,
        layers_to_crop=[4],  # 4 crops the last resnet layer, 3 crops the 3rd, ...etc
        # ---- Aggregator
        # agg_arch='CosPlace',
        # agg_config={'in_dim': 2048,
        #             'out_dim': 2048},
        # agg_arch='GeM',
        # agg_config={'p': 3},
        # agg_arch='ConvAP',
        # agg_config={'in_channels': 2048,
        #             'out_channels': 2048},
        agg_arch="MixVPR",
        agg_config={
            "in_channels": 1024,
            "in_h": 20,
            "in_w": 20,
            "out_channels": 1024,
            "mix_depth": 4,
            "mlp_ratio": 1,
            "out_rows": 4,
        },  # the output dim will be (out_rows * out_channels)
        # ---- Train hyperparameters
        lr=1e-5,  # 0.0002 for adam, 0.05 or sgd (needs to change according to batch size)
        optimizer="adamw",  # sgd, adamw
        weight_decay=0,  # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        warmpup_steps=1000,
        milestones=[5, 10, 15, 25, 45],
        lr_mult=0.01,
        # ----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name="MultiSimilarityLoss",
        miner_name="MultiSimilarityMiner",  # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False,
    )
    train(**config)