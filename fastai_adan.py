from fastai.vision.all import *

# from fastxtend (Benjamin Warner)

def avg_grad(p, beta1, grad_avg=None, **kwargs):
    "Tracks average gradients (m) of `p` in `state` with `beta1`."
    if grad_avg is None: 
        grad_avg = torch.zeros_like(p.grad.data)
    grad_avg.mul_(beta1).add_(p.grad.data, alpha=1-beta1)
    return {'grad_avg': grad_avg}

avg_grad.defaults = dict(beta1=1-0.02)

def avg_diff_grad(p, beta2, prior_grad=None, diff_avg=None, **kwargs):
    "Tracks the average difference of current and prior gradients (v) of `p` in `state` with `beta2`."
    if diff_avg is None: 
        diff_avg   = torch.zeros_like(p.grad.data)
        prior_grad = torch.zeros_like(p.grad.data)

    diff_avg.mul_(beta2).add_(p.grad.data-prior_grad, alpha=1-beta2)
    return {'diff_avg': diff_avg}

avg_diff_grad.defaults = dict(beta2=1-0.08)

def avg_nesterov_est(p, beta2, beta3, prior_grad=None, nesterov_est=None, **kwargs):
    "Tracks the Nesterov momentum estimate of gradients (n) of `p` in `state` with `beta2` & `beta3`."
    if nesterov_est is None: 
        nesterov_est = torch.zeros_like(p.grad.data)
        prior_grad   = torch.zeros_like(p.grad.data)

    nesterov_est.mul_(beta3).add_(torch.square(torch.add(p.grad.data, torch.sub(p.grad.data, prior_grad), alpha=beta2)), alpha=1-beta3)
    return {'nesterov_est': nesterov_est}

avg_nesterov_est.defaults = dict(beta2=1-0.08, beta3=1-0.01)

def prior_grad(p, **kwargs):
    "Register the current gradient of `p` for use in the next step"
    return {'prior_grad' : p.grad.data.clone()}

def debias(beta, step): 
    return 1-beta**step

def adan_step(p, lr, eps, wd, beta1, beta2, beta3, step, grad_avg, diff_avg, nesterov_est, **kwargs):
    "Performs the Adan step with `lr` on `p`"
    db1 = debias(beta1, step)
    db2 = debias(beta2, step)
    db3 = debias(beta3, step)
    wd = (1+lr*wd) if wd!=0 else 1
    lr = lr/torch.sqrt(nesterov_est/db3+eps)
    p.data.sub_(torch.add(grad_avg/db1, diff_avg/db2, alpha=beta2).mul_(lr)).div_(wd)
    return p

def FastAdan(params, lr, beta1=1-0.02, beta2=1-0.08, beta3=1-0.01, eps=1e-8, wd=0.02):
    "A `Optimizer` for Adan with `lr`, `beta`s, `eps` and `params`"
    cbs = [avg_grad, avg_diff_grad, avg_nesterov_est, prior_grad, step_stat, adan_step]
    return Optimizer(params, cbs, lr=lr, beta1=beta1, beta2=beta2, beta3=beta3, eps=eps, wd=wd)
