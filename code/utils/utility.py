import os
import math
import time
import cv2
import datetime
from multiprocessing import Process
from multiprocessing import Queue
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
import numpy as np
import imageio
import skimage
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lrs


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


class checkpoint():
    def __init__(self, args):
        self.args = args
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'model', args.save)
        else:
            self.dir = os.path.join('..', 'model', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        os.makedirs(self.get_path('results'), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt')) else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'F1 score'
        fig = plt.figure()
        plt.title(label)
        plt.plot(axis, self.log[:, 0, 0].numpy(), label='test accuracy per epoch')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('F1')
        plt.grid(True)
        plt.savefig(self.get_path('test.pdf'))
        plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    if tensor.shape[-1] == 1:
                        cv2.imwrite(filename, tensor.numpy())
                    else:
                        imageio.imwrite(filename, tensor.numpy())

        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)]

        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, filename, save_list):
        if self.args.save_results:
            filename = self.get_path('results', '{}_'.format(filename))
            postfix = ('Enh', 'Comp', 'Prob', 'Raw')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_ssim(sr, hr, align=False):
    if align == True:
        sr = sr[:, :, 1:-1, 1:]
        hr = hr[:, :, 2:, 0:-1]
    else:
        sr = sr[:, :, 2:, 0:-1]
        hr = hr[:, :, 2:, 0:-1]
    sr = np.transpose(sr[0].cpu().numpy(), (1, 2, 0))
    hr = np.transpose(hr[0].cpu().numpy(), (1, 2, 0))
    return skimage.measure.compare_ssim(sr / 255., hr / 255., multichannel=True)


def calc_dice(sr, hr, npf=False):
    if not npf:
        sr = np.transpose(sr[0].cpu().numpy(), (1, 2, 0)) / 255.
    hr = np.transpose(hr[0].cpu().numpy(), (1, 2, 0)) / 255.
    dice = np.sum(sr[hr == 1]) * 2.0 / (np.sum(sr) + np.sum(hr))
    return dice


def calculate_Accuracy(confusion):
    confusion = np.asarray(confusion)
    pos = np.sum(confusion, 1).astype(np.float32)  # 1 for row
    res = np.sum(confusion, 0).astype(np.float32)  # 0 for coloum
    tp = np.diag(confusion).astype(np.float32)
    IU = tp / (pos + res - tp)

    meanIU = np.mean(IU)
    Acc = np.sum(tp) / np.sum(confusion)
    Se = confusion[1][1] / (confusion[1][1] + confusion[0][1])
    Sp = confusion[0][0] / (confusion[0][0] + confusion[1][0])

    return meanIU, Acc, Se, Sp, IU


def calc_metrics(pred, sr, hr, npf=False):
    if not npf:
        sr = np.transpose(sr[0].cpu().numpy(), (1, 2, 0)) / 255.
    hr = np.transpose(hr[0].cpu().numpy(), (1, 2, 0)) / 255.
    pred = np.transpose(pred[0].cpu().numpy(), (1, 2, 0))
    sr = sr.reshape([-1]).astype(np.uint8)
    hr = hr.reshape([-1]).astype(np.uint8)
    pred = pred.reshape([-1])
    my_confusion = metrics.confusion_matrix(sr, hr).astype(np.float32)
    meanIU, Acc, Se, Sp, IU = calculate_Accuracy(my_confusion)
    Auc = roc_auc_score(hr, pred)
    return Acc, Se, Sp, Auc, IU[0], IU[1]


def calc_psnr(sr, hr, scale, rgb_range, align=False, dataset=None):
    if hr.nelement() == 1: return 0
    if align == True:  # shift one pixel offset
        sr = sr[:, :, 1:-1, 1:]
        hr = hr[:, :, 2:, 0:-1]
    else:
        sr = sr[:, :, 2:, 0:-1]
        hr = hr[:, :, 2:, 0:-1]
    diff = (sr - hr) / rgb_range
    mse = diff.pow(2).mean()
    return -10 * math.log10(mse)


def image_resize(image, x, y):
    temp = cv2.resize(image, (x, y))
    return temp


def image_cb(alpha, beta, image):
    '''
    g(x) = alpha * f(x) + beta
    '''
    blank = np.zeros(image.shape, image.dtype)
    dst = cv2.addWeighted(image, alpha, blank, 1 - alpha, beta)
    return dst


def visualize(est, label, npf=False):
    if not npf:
        est = np.transpose(est[0].cpu().numpy(), (1, 2, 0))
    est_in = est[:, :, 0]
    label = np.transpose(label[0].cpu().numpy(), (1, 2, 0))[:, :, 0]
    est_color = np.tile(est, (1, 1, 3))  # 3 channel
    est_color[np.logical_and(est_in == 0, label == 255), :] = [255, 0, 0]  # red: false negative
    est_color[np.logical_and(est_in == 255, label == 0), :] = [0, 0, 255]  # green: false positive
    return torch.tensor(np.expand_dims(np.transpose(est_color, (2, 0, 1)), 0))


def visualize_dmap(est, label):
    BCE_loss = F.binary_cross_entropy(est, label, reduction='none')
    pt = torch.exp(-BCE_loss)
    dmap = (1 - pt) ** 2
    return dmap


def calc_boundiou(est, label):
    dmap = visualize_dmap(est, label)
    est = np.transpose(est[0].cpu().numpy(), (1, 2, 0))
    dmap = np.transpose(dmap[0].cpu().numpy(), (1, 2, 0))
    # pmap = np.transpose(pmap[0].cpu().numpy(), (1, 2, 0))
    # total_p = np.sum(pmap)
    # print(total_p)
    s = np.sum(dmap)
    # print(s)
    return float(s)


def make_optimizer(args, target):
    '''make optimizer and scheduler together'''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_last_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

