from abc import abstractmethod, ABCMeta


class Trainer(ABCMeta):

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def train_one_epoch(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def train(self):
        pass
