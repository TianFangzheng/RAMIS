from decimal import Decimal
import utils.utility as utility
import numpy as np
import torch
import torch.nn.utils as utils

torch.autograd.set_detect_anomaly(True)

class Trainer():
    def __init__(self, args, train_loader, eval_loader, my_model, my_loss, ckp):
        self.args = args
        self.ckp = ckp
        self.loader_train = train_loader
        self.loader_test = eval_loader
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        for batch, batch_data in enumerate(self.loader_train):
            batch_data = {key: val.to(self.args.device)
                          for key, val in batch_data.items() if val is not None}
            timer_data.hold()
            timer_model.tic()
            img = batch_data['rgb']
            gt = batch_data['gt']
            self.optimizer.zero_grad()
            # enh, estimation = self.model(img)
            init_seg, implicit, implicit_split, distillation, final_seg = self.model(img)
            # print(init_seg.shape)
            # print(init_seg.max())
            # print(final_seg.shape)
            # print(final_seg.max())
            # print(gt.shape)
            # print(gt.max())
            # exit()

            # loss = self.loss(init_seg, gt) + self.loss(final_seg, gt)
            loss = self.loss(final_seg, gt)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(self.model.parameters(), self.args.gclip)
            self.optimizer.step()
            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()


    def test(self):
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), 1))
        self.model.eval()

        Background_IOU = []
        Vessel_IOU = []
        ACC = []
        SE = []
        SP = []
        AUC = []
        BIOU = []

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()

        for idx_data, batch_data in enumerate(self.loader_test):
            batch_data = {
                key: val.to(self.args.device)
                for key, val in batch_data.items() if val is not None}
            img = batch_data['rgb']
            # enhance, est_o = self.model(img)
            enhance, _, _, _, est_o = self.model(img)
            ve = batch_data['gt']
            est_o = est_o
            ve = ve
            enhance = enhance * 255
            est = est_o * 255.
            hr = img * 255
            ve = ve * 255.
            est[est > 100] = 255
            est[est <= 100] = 0

            est = utility.quantize(est, self.args.rgb_range)
            vis_vessel = utility.visualize(est, ve, False)
            save_list = [enhance, vis_vessel, est_o * 255]
            if self.args.save_gt:
                save_list.extend([hr])
            if self.args.save_results:
                self.ckp.save_results(str(idx_data), save_list)

            # Computing Scores

            self.ckp.log[-1, idx_data, 0] += utility.calc_dice(est, ve, False)
            Acc, Se, Sp, Auc, IU0, IU1 = utility.calc_metrics(est_o, est, ve, False)
            BIOU.append(utility.calc_boundiou(est_o, ve / 255.))
            AUC.append(Auc)
            Background_IOU.append(IU0)
            Vessel_IOU.append(IU1)
            ACC.append(Acc)
            SE.append(Se)
            SP.append(Sp)


        print(np.mean(np.stack(BIOU)))
        print('Acc: %s  |  Se: %s |  Sp: %s |  Auc: %s |  Background_IOU: %s |  vessel_IOU: %s ' % (
        str(np.mean(np.stack(ACC))), str(np.mean(np.stack(SE))), str(np.mean(np.stack(SP))),
        str(np.mean(np.stack(AUC))), str(np.mean(np.stack(Background_IOU))), str(np.mean(np.stack(Vessel_IOU)))))

        self.ckp.log[-1, idx_data, 0] /= 2
        best = self.ckp.log.max(0)
        self.ckp.write_log(
            '[BB xCC]\tDICE Score: {:.6f} (Best: {:.6f} @epoch {})'.format(
                self.ckp.log[-1, idx_data, 0],
                best[0][idx_data, 0],
                best[1][idx_data, 0] + 1))


        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log('Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True)

        torch.set_grad_enabled(True)

    def prepare(self, args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
