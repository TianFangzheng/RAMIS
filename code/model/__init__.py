import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.utils.model_zoo

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')
        self.self_ensemble = args.self_ensemble
        self.precision = args.precision
        self.device = args.device
        self.n_GPUs = args.n_GPUs
        self.gpu_ids = args.gpu_ids
        self.save_models = args.save_models

        module = import_module('model.' + args.model)
        self.model = module.make_model().to(self.device)
        if args.precision == 'half':
            self.model.half()
        self.load(
            ckp.get_path('model'),
            resume=args.resume,
            cpu=False)
        print(self.model, file=ckp.log_file)

    def forward(self, x):
        if self.training:
            if self.n_GPUs > 1:
                return P.data_parallel(self.model, x, device_ids=self.gpu_ids)
            else:
                return self.model(x)
        else:
            forward_function = self.model.forward
            if self.self_ensemble:
                return self.forward_x8(x, forward_function=forward_function)
            else:
                return forward_function(x)

    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]

        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(
                os.path.join(apath, 'model_{}.pt'.format(epoch)))

        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, apath, resume=False, cpu=False):
        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}

        if resume == True:
            load_from = torch.load(os.path.join(apath, 'model_latest.pt'), **kwargs)
        else:
            print('Training from scratch')

        if load_from:
            print('Continue training ...')
            self.model.load_state_dict(load_from, strict=False)


    def forward_x8(self, *args, forward_function=None):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        list_x = []
        for a in args:
            x = [a]
            for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

            list_x.append(x)

        list_y = []
        list_enh = []

        for x in zip(*list_x):
            enh, y = forward_function(*x)
            if not isinstance(y, list):
                y = [y]
                enh = [enh]
            if not list_y:
                list_y = [[_y] for _y in y]
                list_enh = [[_enh] for _enh in enh]

            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)
                for _list_enh, _enh in zip(list_enh, enh): _list_enh.append(_enh)

        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:
                    _list_y[i] = _transform(_list_y[i], 'h')
                if (i % 4) % 2 == 1:
                    _list_y[i] = _transform(_list_y[i], 'v')
        for _list_enh in list_enh:
            for i in range(len(_list_enh)):
                if i > 3:
                    _list_enh[i] = _transform(_list_enh[i], 't')
                if i % 4 > 1:
                    _list_enh[i] = _transform(_list_enh[i], 'h')
                if (i % 4) % 2 == 1:
                    _list_enh[i] = _transform(_list_enh[i], 'v')

        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        enh = [torch.cat(_enh, dim=0).mean(dim=0, keepdim=True) for _enh in list_enh]
        if len(y) == 1: y = y[0]
        if len(enh) == 1: enh = enh[0]
        return enh, y
