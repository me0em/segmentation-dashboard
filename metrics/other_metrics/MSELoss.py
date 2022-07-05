from torch.nn import MSELoss as MSELossParent

class MSELoss(MSELossParent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)