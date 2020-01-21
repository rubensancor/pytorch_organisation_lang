import wandb
from comet_ml import Experiment

# TODO: - Log assets
#       - Log confusion matrix
#       - Log metrics
#       - Log parameters


class Logger():

    def __init__(self, api_key, project_name, args, run_name):
        self.api_key = api_key
        self.project_name = project_name
        self.args = args
        self.run_name = run_name


class Wandb_logger(Logger):

    def __init__(self):
        super(Wandb_logger, self).__init__()

        wandb.init(name=self.run_name,
                   project=self.project_name)


class Comet_logger(Logger):

    def __init__(self, api_key, project_name, args, run_name,workspace, tags):
        super(Comet_logger, self).__init__()

        self.workspace = workspace
        self.tags = tags

        self.experiment = Experiment(api_key=self.api_key,
                                project_name=self.project_name,
                                workspace=self.workspace)
