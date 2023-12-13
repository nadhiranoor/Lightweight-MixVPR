import glob
import os
from typing import Tuple

import torch
from PIL import Image
from torch.utils import data
import numpy as np
import torchvision.transforms as tvf
from tqdm import tqdm
import cv2
import time
import argparse
from main import VPRModel

parser = argparse.ArgumentParser()
# parser.add_argument('--ckpt_path', type=str, default='kd_model/kd_epoch79.ckpt', help='checkpoint path')
parser.add_argument('--model', type=str, default='kd', help='backbone name')
parser.add_argument('--query', type=str, default='/data/VCL/Nadhira/class/datasets/msls_val/test/bengaluru/query/images', help='query image path')
parser.add_argument('--database', type=str, default='/data/VCL/Nadhira/class/datasets/msls_val/test/bengaluru/database/images', help='number of workers')
# parser.add_argument('--save_path', type=str, default='logs/trial1', help='model path saved')
args = parser.parse_args()

class BaseDataset(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache).
    """

    def __init__(self, img_path):
        super().__init__()
        self.img_path = img_path

        # path to images
        if 'query' in self.img_path:
            img_path_list = glob.glob(self.img_path + '/**/**/*.jpg', recursive=True)
            self.img_path_list = img_path_list
        elif 'database' in self.img_path:
            img_path_list = glob.glob(self.img_path + '/**/**/*.jpg', recursive=True)
            # sort images for db
            self.img_path_list = img_path_list #sorted(img_path_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        else:
            raise ValueError('img_path should be either query or db')
        assert len(self.img_path_list) > 0, f'No images found in {self.img_path}'

    def __getitem__(self, index):
        img = load_image(self.img_path_list[index])
        return img, index

    def __len__(self):
        return len(self.img_path_list)


class InferencePipeline:
    def __init__(self, model, dataset, feature_dim, save_path, batch_size=4, num_workers=4, device='cuda'):
        self.model = model
        self.dataset = dataset
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.save_path = save_path

        self.dataloader = data.DataLoader(self.dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          num_workers=self.num_workers,
                                          pin_memory=True,
                                          drop_last=False)

    def run(self, split: str = 'db') -> np.ndarray:

        if os.path.exists(f'{self.save_path}/global_descriptors_{split}.npy'):
            print(f"Skipping {split} features extraction, loading from cache")
            return np.load(f'{self.save_path}/global_descriptors_{split}.npy')

        self.model.to(self.device)
        with torch.no_grad():
            global_descriptors = np.zeros((len(self.dataset), self.feature_dim))
            for batch in tqdm(self.dataloader, ncols=100, desc=f'Extracting {split} features'):
                imgs, indices = batch
                imgs = imgs.to(self.device)

                # model inference
                descriptors = self.model(imgs)
                descriptors = descriptors.detach().cpu().numpy()

                # add to global descriptors
                global_descriptors[np.array(indices), :] = descriptors

        # save global descriptors
        np.save(f'{self.save_path}/global_descriptors_{split}.npy', global_descriptors)
        return global_descriptors


def load_image(path):
    image_pil = Image.open(path).convert("RGB")

    # add transforms
    transforms = tvf.Compose([
        tvf.Resize((320, 320), interpolation=tvf.InterpolationMode.BICUBIC),
        tvf.ToTensor(),
        tvf.Normalize([0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225])
    ])

    # apply transforms
    image_tensor = transforms(image_pil)
    return image_tensor


def load_model(ckpt_path, model_type):
    # Note that images must be resized to 320x320
    
    if model_type=='resnet18':
        model = VPRModel(backbone_arch='resnet18',
                        layers_to_crop=[4],
                        agg_arch='MixVPR',
                        agg_config={'in_channels' : 256,
                                    'in_h' : 20,
                                    'in_w' : 20,
                                    'out_channels' : 128,
                                    'mix_depth' : 4,
                                    'mlp_ratio' : 1,
                                    'out_rows' : 4},
                        )
    
    elif args.model=='efficientnet':
        model = VPRModel(backbone_arch='efficientnet_b0',
                        layers_to_crop=[4],
                        agg_arch='MixVPR',
                        agg_config={'in_channels' : 1280,
                                    'in_h' : 10,
                                    'in_w' : 10,
                                    'out_channels' : 32,
                                    'mix_depth' : 4,
                                    'mlp_ratio' : 1,
                                    'out_rows' : 4},
                        )
        state_dict = torch.load(ckpt_path)
        model.load_state_dict(state_dict['state_dict'])
    elif args.model=='mobilenet':
        model = VPRModel(backbone_arch='mobilenet_v2',
                        layers_to_crop=[4],
                        agg_arch='MixVPR',
                        agg_config={'in_channels' : 512,
                        'in_h' : 20,
                        'in_w' : 20,
                        'out_channels' : 256,
                        'mix_depth' : 4,
                        'mlp_ratio' : 1,
                        'out_rows' : 4},
                        )
        state_dict = torch.load(ckpt_path)
        model.load_state_dict(state_dict['state_dict'])
    elif args.model=='kd': 
        model = VPRModel(backbone_arch='resnet18',
                        layers_to_crop=[4],
                        agg_arch='MixVPR',
                        agg_config={'in_channels' : 256,
                                    'in_h' : 20,
                                    'in_w' : 20,
                                    'out_channels' : 128,
                                    'mix_depth' : 4,
                                    'mlp_ratio' : 1,
                                    'out_rows' : 4},
                        )
        state_dict = torch.load(ckpt_path)
        model.load_state_dict(state_dict)
    else:
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


def calculate_top_k(q_matrix: np.ndarray,
                    db_matrix: np.ndarray,
                    top_k: int = 10) -> np.ndarray:
    # compute similarity matrix
    similarity_matrix = np.matmul(q_matrix, db_matrix.T)  # shape: (num_query, num_db)

    # compute top-k matches
    top_k_matches = np.argsort(-similarity_matrix, axis=1)[:, :top_k]  # shape: (num_query_images, 10)

    return top_k_matches


def record_matches(top_k_matches: np.ndarray,
                   query_dataset: BaseDataset,
                   database_dataset: BaseDataset,
                   out_file: str = 'record.txt') -> None:
    with open(f'{out_file}', 'a') as f:
        for query_index, db_indices in enumerate(tqdm(top_k_matches, ncols=100, desc='Recording matches')):
            pred_query_path = query_dataset.img_path_list[query_index]
            for i in db_indices.tolist():
                pred_db_paths = database_dataset.img_path_list[i]
            f.write(f'{pred_query_path} {pred_db_paths}\n')


def visualize(top_k_matches: np.ndarray,
              query_dataset: BaseDataset,
              database_dataset: BaseDataset,
              visual_dir: str = 'logs/visualize',
              img_resize_size: Tuple = (320, 320)) -> None:
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)
    for q_idx, db_idx in enumerate(tqdm(top_k_matches, ncols=100, desc='Visualizing matches')):
        pred_q_path = query_dataset.img_path_list[q_idx]
        q_array = cv2.imread(pred_q_path, cv2.IMREAD_COLOR)
        q_array = cv2.resize(q_array, img_resize_size, interpolation=cv2.INTER_CUBIC)
        gap_array = np.ones((q_array.shape[0], 10, 3)) * 255  # white gap

        for i in db_idx.tolist():
            pred_db_paths = database_dataset.img_path_list[i]
            db_array = cv2.imread(pred_db_paths, cv2.IMREAD_COLOR)
            db_array = cv2.resize(db_array, img_resize_size, interpolation=cv2.INTER_CUBIC)

            q_array = np.concatenate((q_array, gap_array, db_array), axis=1)

        result_array = q_array.astype(np.uint8)
        # result_array = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)

        # save result as image using cv2
        cv2.imwrite(f'{visual_dir}/{os.path.basename(pred_q_path)}', result_array)


def main():
    # load images
    query_path =  args.query
    datasets_path = args.database
    num_query = len(os.listdir(query_path))
    num_db = len(os.listdir(datasets_path))
    # assert query_path == '' and datasets_path == '', 'Please specify the path to the query and datasets'

    query_dataset = BaseDataset(query_path)
    database_dataset = BaseDataset(datasets_path)

    # load model
    if args.model=='resnet18':
        model_path = '../models/resnet18_epoch(56)_step(13965)_R1[0.8993]_R5[0.9646].ckpt'
        save_path = 'demo/resnet18'
        feature_dim = 512
    elif args.model=='efficientnet':
        model_path = '../models/efficientnet_b0_epoch(73)_step(96422)_R1[0.8148]_R5[0.9295].ckpt'
        save_path = 'demo/efficientnet'
        feature_dim = 128
    elif args.model=='mobilenet':
        model_path = '../models/mobilenet_v2_epoch(77)_step(152412)_R1[0.8850]_R5[0.9627].ckpt'
        save_path = 'demo/mobilenet'
        feature_dim = 1024
    elif args.model=='kd':
        model_path = '../models/kd_epoch79.ckpt'
        save_path = 'demo/kd'
        feature_dim = 512
    elif args.model=='resnet50':
        model_path = '../models/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt'
        save_path = 'demo/resnet50'
        feature_dim=4096
    model = load_model(model_path, args.model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total params: {pytorch_total_params}')

    print(f'number of query images {num_query}')
    print(f'number of database images {num_db}')
    # set up inference pipeline
    database_pipeline = InferencePipeline(model=model, dataset=database_dataset, feature_dim=feature_dim, save_path=save_path)
    query_pipeline = InferencePipeline(model=model, dataset=query_dataset, feature_dim=feature_dim, save_path=save_path)
   
    
    # run inference
    start_inference_db= time.time()
    db_global_descriptors = database_pipeline.run(split='db')  # shape: (num_db, feature_dim)
    end_inferences= time.time()
    print(f'average inference time: {round((end_inferences-start_inference_db)/num_db,5)}s')
    start_inference_query= time.time()
    query_global_descriptors = query_pipeline.run(split='query')  # shape: (num_query, feature_dim)
    end_inferences= time.time()
    print(f'average inference time: {round((end_inferences-start_inference_query)/num_query,5)}s')
    
    # calculate top-k matches
    start_calculate_topk= time.time()
    top_k_matches = calculate_top_k(q_matrix=query_global_descriptors, db_matrix=db_global_descriptors, top_k=10)
    end_calculate_topk= time.time()
    print(f'average record mathces time: {round((end_calculate_topk-start_calculate_topk)/num_query,5)}s')
    
    # record query_database_matches
    start_record_matches= time.time()
    record_matches(top_k_matches, query_dataset, database_dataset, out_file=save_path+'/record.txt')
    end_record_matches= time.time()
    print(f'average record mathces time: {round((end_record_matches-start_record_matches)/num_query,5)}s')
    
    # visualize top-k matches
    visualize(top_k_matches, query_dataset, database_dataset, visual_dir=save_path+'/visualize')


if __name__ == '__main__':
    main()
