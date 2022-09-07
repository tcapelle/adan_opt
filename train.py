from adan_pytorch import Adan
from fastai_adan import FastAdan
from fastai.vision.all import *
from madgrad import MADGRAD

import wandb
from fastai.callback.wandb import WandbCallback

path = untar_data(URLs.IMAGENETTE_160)

def get_learner(img_size=128, arch="convnext_tiny", opt_func=Adam):
    dls = ImageDataLoaders.from_folder(path, valid='val', 
        item_tfms=RandomResizedCrop(img_size, min_scale=0.35), batch_tfms=Normalize.from_stats(*imagenet_stats))
    return vision_learner(dls, arch, metrics=accuracy, pretrained=False, opt_func=opt_func).to_fp16()



def fit(config, group="Adam"):
    with wandb.init(project="adan_optimizer", group=group, config=config):
        learn = get_learner(config.img_size, config.arch, config.opt_func)
        learn.fit_one_cycle(config.epochs, config.lr, wd=config.wd, cbs=WandbCallback(log_preds=False))

adan_lucidrains = partial(OptimWrapper, opt=partial(Adan, betas=(0.02, 0.08, 0.01)))
madgrad = partial(OptimWrapper)

def get_config(opt, arch="convnext_tiny"):
    if opt == "SGD":
        config = SimpleNamespace(lr=1e-2, wd=2e-2, epochs=20, arch=arch, img_size=128, opt_func=partial(SGD, mom=0.9))
    elif opt == "Adan":
        config = SimpleNamespace(lr=1e-2, wd=2e-2, epochs=20, arch=arch, img_size=128, opt_func=adan_lucidrains)
    elif opt == "AdanFastai":
        config = SimpleNamespace(lr=1e-2, wd=2e-2, epochs=20, arch=arch, img_size=128, opt_func=FastAdan)
    elif opt == "MadGrad":
        config = SimpleNamespace(lr=1e-2, wd=2e-2, epochs=20, arch=arch, img_size=128, opt_func=madgrad)
    else:
        config = SimpleNamespace(lr=1e-3, wd=1e-3, epochs=20, arch=arch, img_size=128, opt_func=Adam)
    return config

@call_parse
def run(opt: Param("Optimizer to use", str) = "Adam",
        n: Param("Repeat n times", int) = 1,
        arch: Param("Any timm image backbone", str) = "convnext_tiny",
       ):
    for _ in range(n):
        config = get_config(opt, arch)
        fit(config, opt)
        
    
