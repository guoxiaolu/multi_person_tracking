import cv2
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from .dgnet.trainer import DGNet_Trainer, to_gray
from .dgnet.utils import get_config
from .dgnet.np_distance import compute_dist
from .base import run_time

def normalize(nparray, order=2, axis=0):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)

def GetDis(v0, v1):
    cosine_distance = compute_dist(v0,v1)[0][0]
    return cosine_distance

class ReID:
    def __init__(self, config_name, checkpoint_gen, checkpoint_id, trainer_pth_name, trainer_config_name):
        # Load experiment setting
        config = get_config(config_name)
        # we use config
        config['apex'] = False
        trainer = DGNet_Trainer(config, trainer_pth_name, trainer_config_name)    
        if torch.cuda.is_available():
            state_dict_gen = torch.load(checkpoint_gen)
        else:
            state_dict_gen = torch.load(checkpoint_gen, map_location=torch.device('cpu'))
        trainer.gen_a.load_state_dict(state_dict_gen['a'], strict=False)
        trainer.gen_b = trainer.gen_a
        
        if torch.cuda.is_available():
            state_dict_id = torch.load(checkpoint_id)
        else:
            state_dict_id = torch.load(checkpoint_id, map_location=torch.device('cpu'))
        trainer.id_a.load_state_dict(state_dict_id['a'])
        trainer.id_b = trainer.id_a
        
        if torch.cuda.is_available():
            trainer.cuda()
        trainer.eval()
        self.id_encode = trainer.id_a # encode function
         
        self.data_transforms = transforms.Compose([
            transforms.Resize((config['crop_image_height'], config['crop_image_width']), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    @run_time
    def GetFeature(self,img):
        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_rgb,mode='RGB')
        id_img = self.data_transforms(image)
        with torch.no_grad():
            if torch.cuda.is_available():
                id_img = id_img.cuda()
            id_imgs=id_img.unsqueeze(0)
            fea, _ = self.id_encode(id_imgs)
            fea=fea.cpu().numpy()
            fea = normalize(fea, axis=1)
            return fea

if __name__ == '__main__':
    checkpoint_gen = '/Users/guoxiaolu/work/tracking/models/gen_00100000.pt'
    checkpoint_id = '/Users/guoxiaolu/work/tracking/models/id_00100000.pt'
    config_name = '/Users/guoxiaolu/work/tracking/models/config.yaml'
    im0=cv2.imread('/Users/guoxiaolu/work/PersonTrack/lib/ReID/EANet/data/01.png')
    im1=cv2.imread('/Users/guoxiaolu/work/PersonTrack/lib/ReID/EANet/data/03.png')
    reid=ReID(config_name, checkpoint_gen, checkpoint_id)
    v0=reid.GetFeature(im0)
    v1=reid.GetFeature(im1)
    dis = GetDis(v0, v1)
    print(dis)