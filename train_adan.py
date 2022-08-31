from adan_pytorch import Adan
from fastai_adan import Adan as AdanFastai
from fastai.vision.all import *

import wandb
from fastai.callback.wandb import WandbCallback

path = untar_data(URLs.IMAGENETTE_160)

config = SimpleNamespace(lr=1e-2, 
                         beta1=0.02,
                         beta2=0.08,
                         beta3=0.01,
                         wd=2e-2, 
                         epochs=20, 
                         arch="convnext_tiny", 
                         img_size=128)

def setup_adan(beta1, beta2, beta3, wd):
    # setup optimizer
    return partial(OptimWrapper, opt=partial(Adan, betas=(beta1, beta2, beta3), weight_decay=wd))
    
def get_learner(img_size, arch, opt_func):
    # get dataloaders
    dls = ImageDataLoaders.from_folder(path, valid='val', 
        item_tfms=RandomResizedCrop(img_size, min_scale=0.35), batch_tfms=Normalize.from_stats(*imagenet_stats))
    return vision_learner(dls, arch, metrics=accuracy, pretrained=False, opt_func=opt_func).to_fp16()

def fit(config, group="Adan_sweep"):
    with wandb.init(project="adan_optimizer", group=group, config=config):
        config = wandb.config
        opt_func = setup_adan(config.beta1, config.beta2, config.beta3, config.wd)
        learn = get_learner(config.img_size, config.arch, opt_func)
        learn.fit_one_cycle(config.epochs, config.lr, cbs=WandbCallback(log_preds=False))

@call_parse
def run(
    lr: Param("Adan lr", str) = config.lr,
    beta1: Param("Adan beta1", str) = config.beta1,
    beta2: Param("Adan beta2", str) = config.beta2,
    beta3: Param("Adan beta3", str) = config.beta3,
):
    config.lr = lr
    config.beta1 = beta1
    config.beta2 = beta2
    config.beta3 = beta3
    
    # run the training
    fit(config)
        
if __name__=="__main__":
    print("Training Adan FTW!")
    run()
        
    