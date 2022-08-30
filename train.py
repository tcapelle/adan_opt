from adan_pytorch import Adan
from fastai_adan import Adan as AdanFastai
from fastai.vision.all import *

import wandb
from fastai.callback.wandb import WandbCallback

path = untar_data(URLs.IMAGENETTE_160)

def get_learner(img_size=128, arch="convnext_tiny", opt_func=Adam):
    dls = ImageDataLoaders.from_folder(path, valid='val', 
        item_tfms=RandomResizedCrop(img_size, min_scale=0.35), batch_tfms=Normalize.from_stats(*imagenet_stats))
    return vision_learner(dls, arch, metrics=accuracy, pretrained=False, opt_func=opt_func).to_fp16()

adam_config = SimpleNamespace(lr=1e-3, wd=1e-3, epochs=20, arch="convnext_tiny", img_size=128, opt_func=Adam)

def fit(config, group="Adam"):
    with wandb.init(project="adan_optimizer", group=group, config=config):
        learn = get_learner(config.img_size, config.arch, config.opt_func)
        learn.fit_one_cycle(config.epochs, config.lr, wd=config.wd, cbs=WandbCallback(log_preds=False))

adan = partial(OptimWrapper, opt=partial(Adan, betas=(0.02, 0.08, 0.01)))
adan_config = SimpleNamespace(lr=1e-2, wd=2e-2, epochs=20, arch="convnext_tiny", img_size=128, opt_func=adan)

@call_parse
def run(opt: Param("Optimizer to use", str) = "Adam",
        n: Param("Repeat n times", int) = 1
       ):
    for _ in range(n):
        if opt == "Adam":
            fit(adam_config, group="Adam")
        if opt == "Adan":
            fit(adan_config, group="Adan")
        if opt == "AdanFastai":
            adan_config.opt_func = AdanFastai
            fit(adan_config, group="Adan_fastai")
        
if __name__=="__main__":
    run()
        
    