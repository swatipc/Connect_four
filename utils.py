class Config(dict):
    def __getattr__(self, name):
        return self[name]


# Ref: https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AvgMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count
