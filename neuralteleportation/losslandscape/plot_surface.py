"""
    Authors: Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein.
    Title: Visualizing the Loss Landscape of Neural Nets. NIPS, 2018.
    Source Code: https://github.com/tomgoldstein/loss-landscape

    Modified: Philippe Spino
"""
import argparse
import copy
import h5py
import torch
import time
import socket
import os
import sys
import numpy as np
import torchvision
import torch.nn as nn

import neuralteleportation.losslandscape.scheduler as scheduler
import neuralteleportation.losslandscape.net_plotter as net_plotter
import neuralteleportation.losslandscape.evaluation as evaluation
import neuralteleportation.losslandscape.h5_util as h5_util
import neuralteleportation.losslandscape.plot_2D as plot_2D
import neuralteleportation.losslandscape.plot_1D as plot_1D


class SurfacePlotter():
    """
        This class serve as a holder for the loss landscape surface.
    """
    def __init__(self, net, device, x, y, surf_file=None, directions_file=None, direction_type='weights', same_direction=False, raw_data=True, data_split=1):
        self.net = net
        self.direction_type = direction_type
        self.same_direction = same_direction
        # self.idx = 0
        self.raw_data = raw_data
        self.data_split = data_split
        self.xnorm = 'filter'
        self.ynorm = 'filter'
        self.xignore = 'biasbn'
        self.yignore = 'biasbn'
        self.x = x
        self.y = y

        self.use_cuda = True if device.type == 'cuda' else False

        try:
            self.xmin, self.xmax, self.xnum = [int(a) for a in self.x.split(':')]
            self.ymin, self.ymax, self.ynum = [int(a) for a in self.y.split(':')]
        except:
            raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')

        if not directions_file:
            self.directions_file = self.__name_direction_file__()
        else:
            self.directions_file = self.directions_file
        self.__setup_direction__(self.directions_file, self.net)

        if not surf_file:
            self.surf_file = self.__name_surface_file__()
        else:
            self.surf_file = surf_file
        self.__setup_surface_file__(self.surf_file, self.directions_file)

    def __name_direction_file__(self):
        """
            Name the direction file that stores the random directions.
        """
        dir_file = "./tmp/direction_vector"

        # name for xdirection
        dir_file += '_' + self.direction_type
        if self.xignore:
            dir_file += '_xignore=' + self.xignore
        if self.xnorm:
            dir_file += '_xnorm=' + self.xnorm

        # name for ydirection
        if self.yignore:
            dir_file += '_yignore=' + self.yignore
        if self.ynorm:
            dir_file += '_ynorm=' + self.ynorm
        if self.same_direction:  # ydirection is the same as xdirection
            dir_file += '_same_dir'

        # index number
        # if self.idx > 0:
        #     dir_file += '_idx=' + str(self.idx)

        dir_file += ".h5"

        return dir_file

    def __setup_direction__(self, dir_file, net):
        """
            Setup the h5 file to store the directions.
            - xdirection, ydirection: The pertubation direction added to the mdoel.
            The direction is a list of tensors.
        """
        print('-------------------------------------------------------------------')
        print('setup_direction')
        print('-------------------------------------------------------------------')
        # Skip if the direction file already exists
        if os.path.exists(dir_file):
            f = h5py.File(dir_file, 'r')
            if (self.y and 'ydirection' in f.keys()) or 'xdirection' in f.keys():
                f.close()
                print("%s is already set up" % dir_file)
                return
            f.close()

        # Create the plotting directions
        f = h5py.File(dir_file, 'w')  # create file, fail if exists
        print("Setting up the plotting directions...")
        xdirection = net_plotter.create_random_direction(self.net, self.direction_type, self.xignore, self.xnorm)
        h5_util.write_list(f, 'xdirection', xdirection)

        if self.same_direction:
            ydirection = xdirection
        else:
            ydirection = net_plotter.create_random_direction(net, self.direction_type, self.yignore, self.ynorm)
        h5_util.write_list(f, 'ydirection', ydirection)

        f.close()
        print("direction file created: %s" % dir_file)

    def __name_surface_file__(self):
        # use args.dir_file as the perfix
        surf_file = "./tmp/surface"

        # resolution
        surf_file += '_[%s,%s,%d]' % (str(self.xmin), str(self.xmax), int(self.xnum))
        if self.y:
            surf_file += 'x[%s,%s,%d]' % (str(self.ymin), str(self.ymax), int(self.ynum))

        # dataloder parameters
        if self.raw_data:  # without data normalization
            surf_file += '_rawdata'
        if self.data_split > 1:
            surf_file += '_datasplit=' + str(args.data_split) + '_splitidx=' + str(args.split_idx)

        return surf_file + ".h5"

    def __setup_surface_file__(self, surf_file, directions_file):
        # skip if the direction file already exists
        if os.path.exists(surf_file):
            f = h5py.File(surf_file, 'r')
            if (self.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
                f.close()
                print("%s is already set up" % surf_file)
                return

        f = h5py.File(surf_file, 'a')
        f['directions_file'] = directions_file

        # Create the coordinates(resolutions) at which the function is evaluated
        xcoordinates = np.linspace(self.xmin, self.xmax, num=self.xnum)
        f['xcoordinates'] = xcoordinates

        if self.y:
            ycoordinates = np.linspace(self.ymin, self.ymax, num=self.ynum)
            f['ycoordinates'] = ycoordinates
        f.close()

        return surf_file

    def crunch(self, net, criterion, w, s, dataloader, loss_key, acc_key):
        """
            Calculate the loss values and accuracies of modified models in parallel
            using MPI reduce.
        """

        d = net_plotter.load_directions(self.directions_file)
        rank = 0

        f = h5py.File(self.surf_file, 'r+')
        losses, accuracies = [], []
        xcoordinates = f['xcoordinates'][:]
        ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

        if loss_key not in f.keys():
            shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates), len(ycoordinates))
            losses = -np.ones(shape=shape)
            accuracies = -np.ones(shape=shape)
            if rank == 0:
                f[loss_key] = losses
                f[acc_key] = accuracies
        else:
            losses = f[loss_key][:]
            accuracies = f[acc_key][:]
        f.close()

        # Generate a list of indices of 'losses' that need to be filled in.
        # The coordinates of each unfilled index (with respect to the direction vectors
        # stored in 'd') are stored in 'coords'.
        inds, coords, _ = scheduler.get_job_indices(losses, xcoordinates, ycoordinates, rank=rank)

        print('Computing %d values for rank %d' % (len(inds), rank))
        start_time = time.time()
        total_sync = 0.0

        # Loop over all uncalculated loss values
        for count, ind in enumerate(inds):
            # Get the coordinates of the loss value being calculated
            coord = coords[count]

            # Load the weights corresponding to those coordinates into the net
            if self.direction_type == 'weights':
                net_plotter.set_weights(net, w, d, coord)
            elif args.dir_type == 'states':
                net_plotter.set_states(net.module if args.ngpu > 1 else net, s, d, coord)

            # Record the time to compute the loss value
            loss_start = time.time()
            loss, acc = evaluation.eval_loss(net, criterion, dataloader, use_cuda=self.use_cuda)
            loss_compute_time = time.time() - loss_start

            # Record the result in the local array
            losses.ravel()[ind] = loss
            accuracies.ravel()[ind] = acc

            # Send updated plot data to the master node
            syc_start = time.time()
            syc_time = time.time() - syc_start
            total_sync += syc_time

            print('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f' % (
                    rank, count, len(inds), 100.0 * count / len(inds), str(coord), loss_key, loss,
                    acc_key, acc, loss_compute_time, syc_time))

            # Periodically write to file, and always write after last update
            if count % 90 == 6 * rank or count == len(inds) - 1:
                print('Writing to file')
                f = h5py.File(self.surf_file, 'r+')
                f[loss_key][losses != -1] = losses[losses != -1]
                f[acc_key][accuracies != -1] = accuracies[accuracies != -1]
                f.flush()
                f.close()

        total_time = time.time() - start_time
        print('Rank %d done!  Total time: %.2f Sync: %.2f' % (rank, total_time, total_sync))

        f.close()

    def plot_surface(self, vmin, vmax, vlevel, loss_max, log):
        plot_2D.plot_2d_contour(self.surf_file, 'train_loss', vmin, vmax, vlevel, True)
        plot_1D.plot_1d_loss_err(self.surf_file, self.xmin, self.xmax, loss_max, log, True)


###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
    parser.add_argument('--rank', type=int, default=0, help='The rank of this job when multiple jobs are working together')
    parser.add_argument('--of', type=int, default=1, help='The total number of jobs/ranks that are running')
    parser.add_argument('--cuda', '-c', action='store_true', help='use cuda')
    parser.add_argument('--threads', default=2, type=int, help='number of threads')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use for each rank, useful for data parallel evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')

    # data parameters
    parser.add_argument('--dataset', default='cifar10', help='cifar10 | imagenet')
    parser.add_argument('--datapath', default='cifar10/data', metavar='DIR', help='path to the dataset')
    parser.add_argument('--raw_data', action='store_true', default=False, help='no data preprocessing')
    parser.add_argument('--data_split', default=1, type=int, help='the number of splits for the dataloader')
    parser.add_argument('--split_idx', default=0, type=int, help='the index of data splits for the dataloader')
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')

    # model parameters
    parser.add_argument('--model', default='resnet56', help='model name')
    parser.add_argument('--model_folder', default='', help='the common folder that contains model_file and model_file2')
    parser.add_argument('--model_file', default='', help='path to the trained model file')
    parser.add_argument('--model_file2', default='', help='use (model_file2 - model_file) as the xdirection')
    parser.add_argument('--model_file3', default='', help='use (model_file3 - model_file) as the ydirection')
    parser.add_argument('--loss_name', '-l', default='crossentropy', help='loss functions: crossentropy | mse')

    # direction parameters
    parser.add_argument('--dir_file', default='', help='specify the name of direction file, or the path to an eisting direction file')
    parser.add_argument('--dir_type', default='weights', help='direction type: weights | states (including BN\'s running_mean/var)')
    parser.add_argument('--x', default='-1:1:51', help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', default='-1:1:51', help='A string with format ymin:ymax:ynum')
    parser.add_argument('--xnorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--ynorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--xignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--yignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--same_dir', action='store_true', default=False, help='use the same random direction for both x-axis and y-axis')
    parser.add_argument('--idx', default=0, type=int, help='the index for the repeatness experiment')
    parser.add_argument('--surf_file', default='', help='customize the name of surface file, could be an existing file.')

    # plot parameters
    parser.add_argument('--proj_file', default='', help='the .h5 file contains projected optimization trajectory.')
    parser.add_argument('--loss_max', default=5, type=float, help='Maximum value to show in 1D plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--show', action='store_true', default=False, help='show plotted figures')
    parser.add_argument('--log', action='store_true', default=False, help='use log scale for loss values')
    parser.add_argument('--plot', action='store_true', default=False, help='plot figures after computation')

    args = parser.parse_args()

    print(type(args))

    torch.manual_seed(123)
    #--------------------------------------------------------------------------
    # Environment setup
    #--------------------------------------------------------------------------
    if args.mpi:
        comm = mpi.setup_MPI()
        rank, nproc = comm.Get_rank(), comm.Get_size()
    else:
        comm, rank, nproc = None, args.rank, args.of

    if rank>=nproc or rank<0 or nproc<=0:
        raise Exception('Invalid values for rank (%s) and nproc (%s)'%(rank,nproc))

    # in case of multiple GPUs per node, set the GPU to use for each rank
    if args.cuda:
        if not torch.cuda.is_available():
            raise Exception('User selected cuda option, but cuda is not available on this machine')
        gpu_count = torch.cuda.device_count()
        if args.ngpu==1:
            torch.cuda.set_device(rank % gpu_count)
            print('Rank %d use GPU %d of %d GPUs on %s' %
                  (rank, torch.cuda.current_device(), gpu_count, socket.gethostname()))
        else:
            print('Rank %d using %d GPUs on %s' %
                 (rank, gpu_count, socket.gethostname()))

    #--------------------------------------------------------------------------
    # Check plotting resolution
    #--------------------------------------------------------------------------
    try:
        args.xmin, args.xmax, args.xnum = [int(a) for a in args.x.split(':')]
        args.ymin, args.ymax, args.ynum = (None, None, None)
        if args.y:
            args.ymin, args.ymax, args.ynum = [int(a) for a in args.y.split(':')]
            assert args.ymin and args.ymax and args.ynum, \
            'You specified some arguments for the y axis, but not all'
    except:
        raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')

    #--------------------------------------------------------------------------
    # Load models and extract parameters
    #--------------------------------------------------------------------------
    net = model_loader.load(args.dataset, args.model, args.model_file)
    w = net_plotter.get_weights(net) # initial parameters
    s = copy.deepcopy(net.state_dict()) # deepcopy since state_dict are references
    if args.ngpu > 1:
        # data parallel with multiple GPUs on a single node
        net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    #--------------------------------------------------------------------------
    # Setup the direction file and the surface file
    #--------------------------------------------------------------------------
    dir_file = net_plotter.name_direction_file(args) # name the direction file
    if rank == 0:
        net_plotter.setup_direction(args, dir_file, net)

    surf_file = name_surface_file(args, dir_file)
    if rank == 0:
        setup_surface_file(args, surf_file, dir_file)

    # wait until master has setup the direction file and surface file
    mpi.barrier(comm)

    # load directions
    d = net_plotter.load_directions(dir_file)
    # calculate the consine similarity of the two directions
    if len(d) == 2 and rank == 0:
        similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
        print('cosine similarity between x-axis and y-axis: %f' % similarity)

    #--------------------------------------------------------------------------
    # Setup dataloader
    #--------------------------------------------------------------------------
    # download CIFAR10 if it does not exit
    if rank == 0 and args.dataset == 'cifar10':
        torchvision.datasets.CIFAR10(root=args.dataset + '/data', train=True, download=True)

    mpi.barrier(comm)

    trainloader, testloader = dataloader.load_dataset(args.dataset, args.datapath,
                                args.batch_size, args.threads, args.raw_data,
                                args.data_split, args.split_idx,
                                args.trainloader, args.testloader)

    #--------------------------------------------------------------------------
    # Start the computation
    #--------------------------------------------------------------------------
    crunch(surf_file, net, w, s, d, trainloader, 'train_loss', 'train_acc', args)
    # crunch(surf_file, net, w, s, d, testloader, 'test_loss', 'test_acc', comm, rank, args)

    #--------------------------------------------------------------------------
    # Plot figures
    #--------------------------------------------------------------------------
    if args.plot and rank == 0:
        if args.y:
            plot_2D.plot_2d_contour(surf_file, 'train_loss', args.vmin, args.vmax, args.vlevel, args.show)
        else:
            plot_1D.plot_1d_loss_err(surf_file, args.xmin, args.xmax, args.loss_max, args.log, args.show)
