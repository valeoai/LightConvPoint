import torch
from lightconvpoint.nn import with_indices_computation_rotation

def get_dataset(base_class):

    # create a dataset class that will inherit from base_class
    class LCPDataset(base_class):

        def __init__(self, *args, **kwargs):

            if "network_function" in kwargs:
                net_func = kwargs["network_function"]
                del kwargs["network_function"]
            else:
                net_func = None

            super().__init__(*args, **kwargs)

            if net_func is not None:
                self.net = net_func()
            else:
                self.net = None


        def download(self):
            super().download()

        def process(self):
            super().process()
            
        @with_indices_computation_rotation
        def __getitem__(self, idx):

            data = super().__getitem__(idx)

            return_dict = {}

            for key, value in data.__dict__.items():
                if value is None:
                    continue
                
                if key=='pos':
                    return_dict['pts'] = data.pos.transpose(0,1)
                    return_dict['pos'] = data.pos.transpose(0,1)
                elif key=='x':
                    return_dict['x'] = data.x.transpose(0,1)
                else:
                    return_dict[key] = value

            if 'x' not in return_dict:
                return_dict['x'] = torch.ones_like(return_dict['pos'])

            # pts = data.pos.transpose(0,1)

            # # test if x 
            # if hasattr(data, 'x') and data.x is not None:
            #     fts = data.x.transpose(0,1)
            # else:
            #     fts = torch.ones_like(pts)

            # lbs = data.y

            # return_dict = {
            #     "pts": pts,
            #     "features": fts,
            #     "target": lbs,
            # }

            return return_dict

    return LCPDataset