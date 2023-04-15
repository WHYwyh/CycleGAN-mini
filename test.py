import os
from arg_parse import start_parse
from dataset import create_dataset
from model import CycleGANModel
from torchvision.utils import save_image
import torch
import pdb

if __name__ == '__main__':
    parser = start_parse()
    parser = CycleGANModel.modify_commandline_options(
        parser)  # get test options
    opt = parser.parse_args()
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.serial_batches = True
    # no flip; comment this line if results on flipped images are needed.
    opt.no_flip = True
    # no visdom display; the test code saves the results to a HTML file.
    opt.display_id = -1
    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    # create a model given opt.model and other options
    model = CycleGANModel(opt)
    # regular setup: load and print networks; create schedulers
    model.setup(opt)

    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        # result_image = torch.cat((model.real_A, model.fake_B), 3)
        result_image = torch.cat((model.real_B, model.fake_A), 3)
        save_filename = 'out%d.jpg' % (i)
        save_path = os.path.join(opt.out_dir, save_filename)
        save_image(result_image, save_path)
