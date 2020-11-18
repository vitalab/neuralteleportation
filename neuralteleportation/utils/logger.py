import configparser
import contextlib
from collections import defaultdict
from pathlib import Path
from time import time, sleep

import numpy as np
import pandas as pd
import visdom
import yaml
from comet_ml import Experiment, OfflineExperiment
from torch.utils.tensorboard import SummaryWriter


class BaseLogger:
    """Base class for the loggers."""

    def add_scalar(self, name, value, step):
        pass

    def add_figure(self, name, fig):
        pass

    def add_text(self, name, text):
        pass

    def add_video(self, name, video, fps):
        pass

    def add_graph(self, name, graph, graph_input):
        pass

    def add_histogram(self, model, step):
        pass

    def log_parameters(self, params_dict):
        # Comet compatibility
        raise NotImplementedError

    def log_metrics(self, metrics_dict, epoch):
        # Comet compatibility
        raise NotImplementedError

    def train(self):
        # Comet compatibility
        raise NotImplementedError

    def validate(self):
        # Comet compatibility
        raise NotImplementedError

    def test(self):
        # Comet compatibility
        raise NotImplementedError

    def flush(self):
        pass


class DiskLogger(BaseLogger):
    """Logger for storing offline on disk and manipulate directly and produce matplotlib plots"""
    def __init__(self, experiment_dir, interval_secs=60):
        self.data_dict = defaultdict(dict)
        self.last_log_time = 0
        self.log_interval = interval_secs
        experiment_path = Path(experiment_dir)
        assert experiment_path.exists()
        self.log_file_path: Path = experiment_path / 'metrics.csv'
        self.hparams_file_path: Path = experiment_path / 'hparams.yml'

        self.prefix = None
        self.params_logged = False

    def make_prefixed_metric_name(self, metric_name):
        if self.prefix is None:
            return metric_name
        else:
            return self.prefix + '_' + metric_name

    def _update(self):
        if time() - self.last_log_time > self.log_interval:
            self.flush()
            self.last_log_time = time()

    def flush(self):
        """Write all the data to disk"""
        # TODO write incrementally (append)
        df = pd.DataFrame.from_dict(self.data_dict)
        df.index.name = 'step'
        df.to_csv(self.log_file_path)

    def add_scalar(self, name, value, step):
        """Save the data in memory

        Will not write to disk instantly. Call flush() to do it. Otherwise, it will be called after a time interval.
        """
        self.data_dict[self.make_prefixed_metric_name(name)][step] = value
        self._update()

    def log_parameters(self, params_dict):
        assert not self.params_logged
        with open(self.hparams_file_path, 'w') as f:
            yaml.dump(params_dict, f)
        self.params_logged = True

    def log_metrics(self, metrics_dict, epoch=0):
        for k, v in metrics_dict.items():
            self.data_dict[self.make_prefixed_metric_name(k)][epoch] = v
        self._update()

    @contextlib.contextmanager
    def train(self):
        self.prefix = 'train'
        yield
        self.prefix = ''

    @contextlib.contextmanager
    def validate(self):
        self.prefix = 'validate'
        yield
        self.prefix = ''

    @contextlib.contextmanager
    def test(self):
        self.prefix = 'test'
        yield
        self.prefix = ''


def init_comet_experiment(comet_config: Path) -> Experiment:
    # Builds an `Experiment` using the content of the `comet` section of the configuration file
    config = configparser.ConfigParser()
    config.read(str(comet_config))
    comet_kwargs = config["comet"]
    is_experiment_online = comet_kwargs.getboolean("online", fallback=True)
    if "online" in comet_kwargs:
        del comet_kwargs["online"]
    if is_experiment_online:
        experiment_cls = Experiment
    else:
        experiment_cls = OfflineExperiment
    return experiment_cls(**comet_kwargs, auto_metric_logging=False)


def test_csv_logger():
    expected_dict = defaultdict(dict, {
        'train_loss': {0: 3.1416, 1: 3.1416, 2: 3.1416},
        'valid_loss': {0: 3.1416, 1: 3.1416, 2: 3.1416},
        'test_loss': {0: 3.1416}
    })

    if Path('test').exists():
        Path('test/metrics.csv').unlink()
        Path('test').rmdir()
    Path('test').mkdir()
    csv_logger = DiskLogger('test', interval_secs=2)
    csv_logger.add_scalar('train_loss', 3.1416, 0)
    assert csv_logger.log_file_path.exists()
    contents = csv_logger.log_file_path.read_text()
    csv_logger.add_scalar('valid_loss', 3.1416, 0)
    assert csv_logger.log_file_path.read_text() == contents  # 2nd write has been "buffered"
    sleep(2)
    for epoch in range(1, 3):
        csv_logger.add_scalar('train_loss', 3.1416, epoch)
        csv_logger.add_scalar('valid_loss', 3.1416, epoch)
    assert csv_logger.log_file_path.read_text() != contents
    csv_logger.add_scalar('test_loss', 3.1416, 0)
    csv_logger.flush()

    assert csv_logger.data_dict == expected_dict
    Path('test/metrics.csv').unlink()
    Path('test').rmdir()


if __name__ == '__main__':
    """Unit tests"""

    test_csv_logger()
