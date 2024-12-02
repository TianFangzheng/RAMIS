import os
import torch
import argparse
import loss
import model
import utils.utility as utility
from utils.trainer import Trainer
from dataloaders.train_data_loader import Train_Data
from dataloaders.evaluate_data_loader import Evaluate

def parse_args():
    parser = argparse.ArgumentParser(description='general_model')
    parser.add_argument('--self_ensemble', default=False, action='store_true', help='use self-ensemble method for test')
    parser.add_argument('--precision', type=str, default='single', choices=('single', 'half'), help='FP precision for test (single | half)')
    parser.add_argument('--batch_size', type=int, default=10, help='number of batch size')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs to train')
    parser.add_argument('--workers', default=4, type=int, help='number of worker threads')
    parser.add_argument('--print_every', type=int, default=2, help='how many batches to wait before logging training status')
    parser.add_argument('--gpu_ids', type=str, default='0', help='number of GPUs')

    parser.add_argument('--loss', type=str, default='1*DICE+1*MSE+1*L1+1*BCE', choices=('MSE', 'L1', 'BCE', 'DICE', 'FL', 'PDICE', 'PLBCE', 'PBCE', 'SBCE', 'SGBCE'), help='loss function configuration')
    parser.add_argument('--train_set', type=str, default='', help='training set')
    parser.add_argument('--test_set', type=str, default='', help='test set')
    parser.add_argument('--model', default='general_net', help='model name')
    parser.add_argument('--load', type=str, default='', help='file name to load--Continue training ...')

    parser.add_argument('--test_only', action='store_true', help='set this option to test the model')
    parser.add_argument('--save_models', action='store_true', help='save all intermediate models')
    parser.add_argument('--cpu', action='store_true', help='use cpu only')
    parser.add_argument('--save_results', default=True, action='store_true', help='save output results')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--save_gt', default=True, action='store_true', help='save low-resolution and high-resolution images together')
    parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
    parser.add_argument('--save', type=str, default='', help='file name to save')

    # Optimization specifications
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--decay', type=str, default='200', help='learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
    parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop'), help='optimizer to use (SGD | ADAM | RMSprop)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='ADAM beta')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--gclip', type=float, default=0, help='gradient clipping threshold (0 = no clipping)')
    parser.add_argument('--skip_threshold', type=float, default='1e8', help='skipping batch that has large error')
    args = parser.parse_args()
    return args


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    global model
    args = parse_args()
    args.device = torch.device("cuda")
    args.gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    args.n_GPUs = len(args.gpu_ids)
    if args.load != '':
        args.resume = True
    else:
        args.resume = False

    train_dataset = Train_Data(args.train_set)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, sampler=None)
    print("==> train_loader size:{}".format(len(train_loader)))
    val_dataset = Evaluate(args.test_set)
    val_loader = torch.utils.data.DataLoader(val_dataset)  # set batch size to be 1 for validation
    print("==> val_loader size:{}".format(len(val_loader)))

    checkpoint = utility.checkpoint(args)
    model_ = model.Model(args, checkpoint)

    loss_ = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, train_loader, val_loader, model_, loss_, checkpoint)

    while not t.terminate():
        t.train()
        t.test()
    checkpoint.done()


if __name__ == '__main__':
    main()
