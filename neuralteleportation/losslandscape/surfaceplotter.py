"""
    Code Author: Philippe Spino

    Strongly inspired by:
        Authors: Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein.
        Title: Visualizing the Loss Landscape of Neural Nets. NIPS, 2018.
        Source Code: https://github.com/tomgoldstein/loss-landscape

    This class serve as a serializer for the surface drawn by the author's code.

    Last Modified: 1 June 2020
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

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.markers as mmarks
from matplotlib import cm

import neuralteleportation.losslandscape.scheduler as scheduler
import neuralteleportation.losslandscape.net_plotter as net_plotter
import neuralteleportation.losslandscape.evaluation as evaluation
import neuralteleportation.losslandscape.h5_util as h5_util
import neuralteleportation.losslandscape.plot_2D as plot_2D
import neuralteleportation.losslandscape.plot_1D as plot_1D
import neuralteleportation.losslandscape.projection as proj


class SurfacePlotter():
    """
        This class serve as a holder for the loss landscape surface.
        It is based on the publication of Visualizing the Loss Landscape of Neural Nets

        Args:
            net_name(string): string to identify the created surface file if none is given.
            net(nn.Modules or NeuralTeleportationModel): the network to be plotted.
            x, y(string): string format of min:max:precision that is used to define the space where the surface will be plotted.
            surf_file(string):  the h5 file path containing a surface to be plotted.
                        If left to None, it will generate a new file for the current network.
            direction_file(string): the h5 file path containing the values of the random direction vector used in the
                            Filter-wise Normalized contour plot.
            direction_type(string): key specifying if the random direction vector should be build using weights or using torch.network_state.
            same_direction(bool): specify if the y directional vector should be the same as the x directional vector.
            raw_data(bool): specify if the data used in train should be left as is.
            data_split(int): how much splits should be done inside the dataloader.

        Example:
            net = nn.Conv2D(1,3,3)
            x = '-1:1:5'
            surfplt = SurfacePlotter('resnet56', net, x, surf_file='')
            surfplt.crunch(criterion, w1, None, trainloader, 'train_loss', 'train_acc', device)
    """
    def __init__(self, net_name, net, x, y=None, surf_file=None, directions_file=None, direction_type='weights', same_direction=False, raw_data=True, data_split=1):
        self.net_name = net_name
        self.net = net
        self.direction_type = direction_type
        self.same_direction = same_direction
        self.raw_data = raw_data
        self.data_split = data_split
        self.x = x
        self.y = y
        self.xnorm = 'filter'
        self.ynorm = 'filter'
        self.xignore = 'biasbn'
        self.yignore = 'biasbn'
        self.surface = None

        try:
            self.xmin, self.xmax, self.xnum = [float(a) for a in self.x.split(':')]
            self.ymin, self.ymax, self.ynum = (None, None, None)
            if y:
                self.ymin, self.ymax, self.ynum = [float(a) for a in self.y.split(':')]
        except ValueError:
            raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')

        if not directions_file:
            self.directions_file = self.__name_direction_file__()
        else:
            self.directions_file = self.directions_file
        self.__setup_direction__()

        if not surf_file:
            self.surf_file = self.__name_surface_file__()
        else:
            self.surf_file = surf_file
        self.__setup_surface_file__()

    def __name_direction_file__(self):
        """
            Name the direction file that stores the random directions base on the net name.
        """
        dir_file = "./tmp/" + self.net_name + "_direction_vector"

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

        dir_file += ".h5"

        return dir_file

    def __setup_direction__(self):
        """
            Setup the h5 file to store the directions.
            - xdirection, ydirection: The pertubation direction added to the model.
            The direction is a list of tensors.
        """
        print('-------------------------------------------------------------------')
        print('setup_direction')
        print('-------------------------------------------------------------------')

        # Open if the direction file already exists or no
        if os.path.exists(self.directions_file):
            f = h5py.File(self.directions_file, 'r')
            if (self.y and 'ydirection' in f.keys()) or 'xdirection' in f.keys():
                f.close()
                print("%s is already set up" % self.directions_file)
                return
            f.close()

        # Create the plotting directions
        f = h5py.File(self.directions_file, 'w')  # create file, fail if exists
        print("Setting up the plotting directions...")
        xdirection = net_plotter.create_random_direction(self.net, self.direction_type, ignore=self.xignore, norm=self.xnorm)
        h5_util.write_list(f, 'xdirection', xdirection)

        if self.y:
            if self.same_direction:
                ydirection = xdirection
            else:
                ydirection = net_plotter.create_random_direction(self.net, self.direction_type, ignore=self.yignore, norm=self.ynorm)
            h5_util.write_list(f, 'ydirection', ydirection)

        f.close()
        print("direction file created: %s" % self.directions_file)

    def __name_surface_file__(self):
        """
            Creates the name of the network the surface file.
        """
        # use self.dir_file as the perfix
        surf_file = "./tmp/" + self.net_name + "_surface"

        # resolution
        surf_file += '_[%s,%s,%d]' % (str(self.xmin), str(self.xmax), int(self.xnum))
        if self.y:
            surf_file += 'x[%s,%s,%d]' % (str(self.ymin), str(self.ymax), int(self.ynum))

        # dataloder parameters
        if self.raw_data:  # without data normalization
            surf_file += '_rawdata'
        if self.data_split > 1:
            surf_file += '_datasplit=' + str(self.data_split) + '_splitidx=' + str(self.split_idx)

        return surf_file + ".h5"

    def __setup_surface_file__(self):
        """
            Setup for the surface h5 file.
        """
        # skip if the direction file already exists
        if os.path.exists(self.surf_file):
            f = h5py.File(self.surf_file, 'r')
            if (self.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
                f.close()
                print("%s is already set up" % self.surf_file)
                return

        print("Setting up the surface file...")
        f = h5py.File(self.surf_file, 'a')
        f['directions_file'] = self.directions_file

        # Create the coordinates(resolutions) at which the function is evaluated
        xcoordinates = np.linspace(self.xmin, self.xmax, num=int(self.xnum))
        f['xcoordinates'] = xcoordinates

        if self.y:
            ycoordinates = np.linspace(self.ymin, self.ymax, num=int(self.ynum))
            f['ycoordinates'] = ycoordinates
        f.close()
        print("surface file created: %s" % self.surf_file)

    def crunch(self, criterion, w, s, dataloader, loss_key, acc_key, device='cpu'):
        """
            Calculate the loss values and accuracies of modified model by replacing the weights 
            with a new set of weights. These are computed off the f(a,b) = L(theta + a*d1 + b*d2)
        """

        d = net_plotter.load_directions(self.directions_file)

        if len(d) == 2:
            similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
            print('cosine similarity between x-axis and y-axis: %f' % similarity)

        f = h5py.File(self.surf_file, 'r+')
        losses, accuracies = [], []
        xcoordinates = f['xcoordinates'][:]
        ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

        if loss_key not in f.keys():
            shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates), len(ycoordinates))
            losses = -np.ones(shape=shape)
            accuracies = -np.ones(shape=shape)
            f[loss_key] = losses
            f[acc_key] = accuracies
        else:
            losses = f[loss_key][:]
            accuracies = f[acc_key][:]

        # Generate a list of indices of 'losses' that need to be filled in.
        # The coordinates of each unfilled index (with respect to the direction vectors
        # stored in 'd') are stored in 'coords'.
        inds, coords, _ = scheduler.get_job_indices(losses, xcoordinates, ycoordinates)

        print('Computing %d values' % (len(inds)))
        start_time = time.time()
        total_sync = 0.0

        # Loop over all uncalculated loss values
        for count, ind in enumerate(inds):
            # Get the coordinates of the loss value being calculated
            coord = coords[count]

            # Load the weights corresponding to those coordinates into the net
            if self.direction_type == 'weights':
                net_plotter.set_weights(self.net, w, d, coord, device)
            elif self.direction_type == 'states':
                net_plotter.set_states(self.net, s, d, coord, device)

            # Record the time to compute the loss value
            loss_start = time.time()
            loss, acc = evaluation.eval_loss(self.net, criterion, dataloader, device=device)
            loss_compute_time = time.time() - loss_start

            # Record the result in the local array
            losses.ravel()[ind] = loss
            accuracies.ravel()[ind] = acc

            # Send updated plot data to the master node
            syc_start = time.time()
            syc_time = time.time() - syc_start
            total_sync += syc_time

            print('Evaluated %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f' % (
                    count + 1, len(inds), 100.0 * count / len(inds), str(coord), loss_key, loss,
                    acc_key, acc, loss_compute_time, syc_time))

            # Periodically write to file, and always write after last update
            if (count + 1) % 25 == 0 or count + 1 == len(inds):
                print('Writing to file')
                f[loss_key][losses != -1] = losses[losses != -1]
                f[acc_key][accuracies != -1] = accuracies[accuracies != -1]
                f.flush()

        total_time = time.time() - start_time
        print('done! Total time: %.2f Sync: %.2f' % (total_time, total_sync))
        f.close()

    def plot_surface(self, additional_points=None, surf_name='train_loss'):
        """
            Plot the computed surface and adds point from given args.

            Args:
                surf_name: name of the plot.
        """
        f = h5py.File(self.surf_file, 'r')
        x = np.array(f['xcoordinates'][:])
        y = np.array(f['ycoordinates'][:])
        X, Y = np.meshgrid(x, y)

        if surf_name in f.keys():
            Z = np.array(f[surf_name][:])
        elif surf_name == 'train_err' or surf_name == 'test_err':
            Z = 100 - np.array(f[surf_name][:])

        fig = plt.figure()
        ax = Axes3D(fig)

        if additional_points:
            for p in additional_points:
                ax.scatter3D(p[0], p[1], p[2], color='Black', marker='o')

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        fig.savefig(self.surf_file + '_' + surf_name + '_3dsurface.pdf', dpi=300,
                    bbox_inches='tight', format='pdf')

        f.close()

    def plot_contours(self, vmin, vmax, vlevel, surf_name='train_loss', save=False):
        """
        """
        f = h5py.File(self.surf_file, 'r')
        x = np.array(f['xcoordinates'][:])
        y = np.array(f['ycoordinates'][:])

        if surf_name in f.keys():
            Z = np.array(f[surf_name][:])
        elif surf_name == 'train_err' or surf_name == 'test_err' :
            Z = 100 - np.array(f[surf_name][:])

        X, Y = np.meshgrid(x, y)

        fig = plt.figure()
        CS = plt.contour(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
        plt.clabel(CS, inline=1, fontsize=8)

        if save:
            fig.savefig(surf_file + '_' + self.surf_name + '_2dcontour' + '.pdf', dpi=300,
                        bbox_inches='tight', format='pdf')

    def show(self):
        plt.show()

    @staticmethod
    def plot_from_surf_file(surf_file):
        """
            a static method to plot an already computed surface.

            Args:
                surf_file(string): the path to the h5 surface file.
        """
        f = h5py.File(surf_file, 'r')
        x = np.array(f['xcoordinates'][:])
        y = np.array(f['ycoordinates'][:])
        X, Y = np.meshgrid(x, y)
        Z = np.array(f['train_loss'][:])

        fig = plt.figure()
        ax = Axes3D(fig)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        f.close()

        plt.show()


if __name__ == '__main__':

    import torchvision.transforms as transforms
    from neuralteleportation.models.model_zoo.resnetcob import resnet50COB
    from neuralteleportation.models.model_zoo.vggcob import vgg11COB
    from neuralteleportation.training.experiment_setup import get_cifar10_datasets
    from neuralteleportation.training.training import train, test
    from neuralteleportation.metrics import accuracy
    from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Parameters of the surface plotter.
    batch_size = 128
    data_split = 1

    trainset, valset, testset = get_cifar10_datasets()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    net = resnet50COB(input_channels=3, num_classes=10).to(device)
    w = net_plotter.get_weights(net)

    surfplt = SurfacePlotter('test', net, x='-1:1:5', y='-1:1:5')
    surfplt.crunch(criterion, w, None, trainloader, 'train_loss', 'train_acc', device)

    surfplt.plot_surface()
    plt.show()
