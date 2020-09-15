import configparser
from collections import defaultdict
from pathlib import Path
from time import time, sleep

import numpy as np
import pandas as pd
import visdom
from comet_ml import Experiment
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

    def flush(self):
        pass


class CsvLogger(BaseLogger):
    """Logger for storing offline on disk and manipulate directly and produce matplotlib plots"""
    def __init__(self, experiment_dir, interval_secs=60):
        self.data_dict = defaultdict(dict)
        self.last_log_time = 0
        self.log_interval = interval_secs
        experiment_path = Path(experiment_dir)
        experiment_path.mkdir()
        self.log_file_path: Path = experiment_path / 'metrics.csv'

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
        self.data_dict[name][step] = value
        self._update()


class VisdomLogger(BaseLogger):
    """Visdom logger to log data to a visdom instance"""

    def __init__(self, env, server="http://localhost", port=8097):
        super(VisdomLogger, self).__init__()
        self.env = env
        self.vis = visdom.Visdom(env=self.env, server=server, port=port)
        self.scalars = {}

    def add_scalar(self, name, value, step):
        """
        Logs a scalar plot to visdom, subsequent calls to add_scalar update the corresponding scalar window.
        Args:
            name: str, Scalar name
            value: float, scalar value
            step: int, step for the logged value
        """
        if name not in self.scalars.keys():
            self.scalars[name] = {"values": [value], "steps": [step]}
            self.scalars[name]["window"] = self.vis.line(
                X=np.asarray(self.scalars[name]["steps"]),
                Y=np.asarray(self.scalars[name]["values"]),
                name=name,
                opts=dict(xlabel="steps", ylabel=name, title=name.title()),
            )
        else:
            self.scalars[name]["values"].append(value)
            self.scalars[name]["steps"].append(step)
            self.vis.line(
                X=np.asarray(self.scalars[name]["steps"]),
                Y=np.asarray(self.scalars[name]["values"]),
                win=self.scalars[name]["window"],
                name=name,
                update="replace",
            )

    def add_figure(self, name, fig):
        self.vis.matplot(fig)

    def add_text(self, name, text):
        self.vis.text(text)


class TensorboardLogger(BaseLogger):
    """Tensorboard logger"""

    def __init__(self, env, log_dir="logs/"):
        super(TensorboardLogger, self).__init__()
        # Ensure both are directories
        self.env = env if env[-1] == "/" else env + "/"
        self.log_dir = log_dir if log_dir[-1] == "/" else log_dir + "/"
        self.writer = SummaryWriter(self.log_dir + self.env)

    def add_scalar(self, name, value, step):
        self.writer.add_scalar(name, value, step)

    def add_figure(self, name, fig):
        self.writer.add_figure(name.title(), fig)

    def add_text(self, name, text):
        self.writer.add_text(name.title(), text)

    def add_video(self, name, video, fps=1):
        self.writer.add_video(name.title(), video, fps=fps)

    def add_graph(self, name, graph, graph_input):
        self.writer.add_graph(graph, graph_input)

    def add_histogram(self, model, step):
        model_params = model.state_dict()
        for k, v in model_params.items():
            grads = v
            name = k
            self.writer.add_histogram(tag=name, values=grads, global_step=step)


class MultiLogger(BaseLogger):
    """Concatenate multiple loggers"""

    def __init__(self, sub_loggers):
        super(MultiLogger, self).__init__()
        self.sub_loggers = sub_loggers
        assert isinstance(self.sub_loggers, list), "sub_loggers must be a list"
        assert len(self.sub_loggers) != 0, "sub_loggers is empty"
        for logger in self.sub_loggers:
            assert isinstance(
                logger, BaseLogger
            ), "sub_loggers must contain only BaseLogger derived loggers"

    def add_scalar(self, name, value, step):
        for logger in self.sub_loggers:
            logger.add_scalar(name, value, step)

    def add_figure(self, name, fig):
        for logger in self.sub_loggers:
            logger.add_figure(name, fig)

    def add_text(self, name, text):
        for logger in self.sub_loggers:
            logger.add_text(name, text)

    def add_video(self, name, video, fps=1):
        for logger in self.sub_loggers:
            logger.add_video(name, video, fps)

    def add_graph(self, name, graph, graph_input):
        for logger in self.sub_loggers:
            logger.add_graph(name, graph, graph_input)

    def add_histogram(self, model, step):
        for logger in self.sub_loggers:
            logger.add_histogram(model, step)


def init_comet_experiment(comet_config: Path) -> Experiment:
    # Builds an `Experiment` using the content of the `comet` section of the configuration file
    config = configparser.ConfigParser()
    config.read(str(comet_config))
    return Experiment(**dict(config["comet"]), auto_metric_logging=False)


def test_csv_logger():
    expected_dict = defaultdict(dict, {
        'train_loss': {0: 3.1416, 1: 3.1416, 2: 3.1416},
        'valid_loss': {0: 3.1416, 1: 3.1416, 2: 3.1416},
        'test_loss': {0: 3.1416}
    })

    if Path('test').exists():
        Path('test/metrics.csv').unlink()
        Path('test').rmdir()
    csv_logger = CsvLogger('test', interval_secs=2)
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
