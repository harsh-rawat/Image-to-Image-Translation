class Loss:
    def __init__(self):
        self.loss = []

    def add(self, x):
        self.loss.extend(x)

    def length(self):
        return len(self.loss)

    def get_loss(self):
        return self.loss;
