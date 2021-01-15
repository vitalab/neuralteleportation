import configparser
import contextlib
from collections import defaultdict
from pathlib import Path
from time import time, sleep

import pandas as pd
import yaml
from comet_ml import Experiment


class BaseLogger:
    """Base class for the loggers."""

    def add_scalar(self, name, value, step):
        pass

    def add_text(self, name, text):
        print(name, text)

    def log_parameters(self, params_dict):
        # Comet compatibility
        pass

    def log_metrics(self, metrics_dict, epoch):
        # Comet compatibility
        pass

    def train(self):
        # Comet compatibility
        pass

    def validate(self):
        # Comet compatibility
        pass

    def test(self):
        # Comet compatibility
        pass

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

    def add_scalar(self, name, value, epoch):
        """Save the data in memory

        Will not write to disk instantly. Call flush() to do it. Otherwise, it will be called after a time interval.
        """
        self.data_dict[self.make_prefixed_metric_name(name)][epoch] = value
        self._update()

    def log_parameters(self, params_dict):
        assert not self.params_logged
        with open(self.hparams_file_path, 'w') as f:
            yaml.dump(params_dict, f)
        self.params_logged = True

    def log_metrics(self, metrics_dict, epoch):
        for k, v in metrics_dict.items():
            self.data_dict[self.make_prefixed_metric_name(k)][epoch] = v
        self._update()

    def set_context_prefix(self, prefix):
        self.prefix = prefix

    def reset_context_prefix(self):
        self.prefix = None

    @contextlib.contextmanager
    def train(self):
        self.prefix = 'train'
        yield
        self.prefix = None

    @contextlib.contextmanager
    def validate(self):
        self.prefix = 'validate'
        yield
        self.prefix = None

    @contextlib.contextmanager
    def test(self):
        self.prefix = 'test'
        yield
        self.prefix = None


class CometLogger(BaseLogger):
    def __init__(self, experiment_id=None):
        self.experiment = Experiment(auto_metric_logging=False)
        if experiment_id is not None:
            self.experiment.log_parameter('experiment_id', experiment_id)

    def add_scalar(self, name, value, step):
        self.experiment.log_metric(name, value, epoch=step)

    def log_parameters(self, params_dict):
        self.experiment.log_parameters(params_dict)

    def log_metrics(self, metrics_dict, epoch):
        self.experiment.log_metrics(metrics_dict, epoch=epoch)

    def add_text(self, name, text):
        self.experiment.log_text(f'{name}: {text}')

    def set_context_prefix(self, prefix):
        self.experiment.context = prefix

    def reset_context_prefix(self):
        self.experiment.context = None


class MultiLogger:
    def __init__(self, loggers):
        self.loggers = loggers

    @contextlib.contextmanager
    def train(self):
        for l in self.loggers:
            l.set_context_prefix('train')
        yield
        for l in self.loggers:
            l.reset_context_prefix()

    @contextlib.contextmanager
    def validate(self):
        for l in self.loggers:
            l.set_context_prefix('validate')
        yield
        for l in self.loggers:
            l.reset_context_prefix()

    @contextlib.contextmanager
    def test(self):
        for l in self.loggers:
            l.set_context_prefix('test')
        yield
        for l in self.loggers:
            l.reset_context_prefix()

    def __getattr__(self, item):
        method = getattr(self.loggers[0], item)
        assert callable(method), f'MultiLogger only supports method access; {item} is not callable.'

        def new_f(*args, **kwargs):
            for l in self.loggers:
                method = getattr(l, item)
                method(*args, **kwargs)
        return new_f


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
