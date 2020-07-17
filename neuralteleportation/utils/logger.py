import numpy as np
# import visdom
from torch.utils.tensorboard import SummaryWriter


class BaseLogger:
    """Base class for the loggers."""

    def __init__(self):
        super(BaseLogger, self).__init__()

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


class VisdomLogger(BaseLogger):
    """Visdom logger to log data to a visdom instance"""

    def __init__(self, env, server="http://localhost", port=8097):
        super(VisdomLogger, self).__init__()
        self.env = env
        # self.vis = visdom.Visdom(env=self.env, server=server, port=port)
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
