import time
from arg_parse import start_parse
from dataset import create_dataset
from model import CycleGANModel
from utils import print_current_losses
import pdb
from tqdm import tqdm

if __name__ == '__main__':
    parser = start_parse()

    opt = CycleGANModel.modify_commandline_options(parser)

    # get training options
    opt = parser.parse_args()
    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    # create a model given opt.model and other options
    model = CycleGANModel(opt)

    # regular setup: load and print networks; create schedulers
    model.setup(opt)
    total_iters = 0                # the total number of training iterations

    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_iter = 0
        # update learning rates in the beginning of every epoch.
        model.update_learning_rate()
        for i, data in enumerate(tqdm(dataset)):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            # unpack data from dataset and apply preprocessing
            model.set_input(data)
            # calculate loss functions, get gradients, update network weights
            model.optimize_parameters()

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' %
                      (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch,
              opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
