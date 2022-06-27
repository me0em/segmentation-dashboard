import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
import importlib
import importlib.util
import sys
import random
import logging
import ntpath
from typing import TypeVar, Generic

Shape = TypeVar("Shape")
DType = TypeVar("DType")

class Array(np.ndarray, Generic[Shape, DType]):
    pass


logger = logging.getLogger("dashboard")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)


class Board():
    """ Use this class to ...
    
    Args:
        dict_images <dict>: images, pairly packed in dict for ...
        ...
    """
    def __init__(
            self,
            dataloaders: list,  # list of different dls with dl.dataset.name attr
            model,
            model_name,
            render_fn,
            render_dl_idx,
            obj_item,
            obj_name_item,
            to_model_item,
            limit=10,
            phs_in_row=4,
            get_randomly=False,
            c=3
        ):
        
        # model vars
        self.dataloaders = dataloaders
        self.model = model
        self.model_name = model_name
        # render vars
        self.render_fn = render_fn
        self.render_dl_idx = render_dl_idx
        self.obj_item = obj_item
        self.obj_name_item = obj_name_item
        self.to_model_item = to_model_item
        self.limit = limit
        self.phs_in_row = phs_in_row
        self.get_randomly = get_randomly
        self.stratch_factor = c

    @staticmethod
    def _cut_module_name(name):
        path, name = ntpath.split(name)
        name = name.replace('.py', "")

        return name

    def _load_metrics(self):
        module_paths = glob.glob("metrics/*.py")
        load_info_gen = ((m, self._cut_module_name(m)) for m in module_paths)
        
        metrics = {}

        for module_path, module_name in load_info_gen:
            logger.info(f"Load {module_name} interface from {module_path}")
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            metric_class = getattr(module, module_name)

            metrics[module_name] = metric_class
            
        return metrics

    @staticmethod
    def _gv(d: dict, ind: int):
        """ Get dict value by index of key 
        """
        key = list(d.keys())[ind]
        
        return d[key]

    def run(self, device="cuda:0"):
        storages = []
        dl_names = []
        
        for dl in self.dataloaders:
            storage = {}
            
            logger.info(f"Try to form a storage with {len(dl)} objects from dataloader. Device: {device}")
            for data in dl:
                data[self.obj_item] = data[self.obj_item].to(device)
                data[self.to_model_item] = data[self.to_model_item].to(device)

                storage[data[self.obj_name_item][0]] = (        # IF BATCH_SIZE=1 WE NEED [0]
                    data[self.obj_item],
                    self.model.predict(data[self.to_model_item]),  # изображение FIXME
                )
                
            storages.append(storage)
            dl_names.append(dl.dataset.name)
        
        
        logger.info("Storages has been formed")
                
        logger.info("Render...") 
        self.render(storages[self.render_dl_idx])
        
        logger.info("Measure...")
        results = self.measure(storages, dl_names)
        
        return results
        
        logger.info("Dump results... / TODO")
        # self.dump_results(storage)

    # def render(self, storage: dict[str, tuple[Array["H,W", float], Array["H,W", float]]]):
    def render(self, storage):
        """ TODO
        """
        storage_keys = list(storage.keys())[:self.limit]
        storage = {k[-9:]:v for k,v in storage.items() if k in storage_keys}  # КОСТЫЛЬ
        
        n = len(storage)
        
        for key,value in storage.items():
            storage[key] = tuple(map(self.render_fn, value))
            
        assert n != 0, "Dataset is empty"  # TODO: think about exceptions if columns more then photos
        
        lines_mod = divmod(2*n, 2*self.phs_in_row)
        lines = lines_mod[0]  # number of rows
        phs_left = lines_mod[1]  # количество фото в неполной строке
        
        if phs_left == 0:
            fig, axs = plt.subplots(
                lines, 2*self.phs_in_row,
                figsize=(self.stratch_factor*2*self.phs_in_row, self.stratch_factor*(lines+1))
            )  # добавить потом условия про остаток
        
        else:
            fig, axs = plt.subplots(
                lines+1, 2*self.phs_in_row,
                figsize=(self.stratch_factor*2*self.phs_in_row, self.stratch_factor*(lines+1))
            )
        
        k = 0
        
        if (lines > 1) or ((lines == 1) and (phs_left != 0)):
            for i in range(lines):
                for j in range(0, 2*self.phs_in_row, 2):
                    axs[i, j].imshow(self._gv(storage, k)[0], cmap="gray")
                    axs[i, j].set_title(
                        list(storage.keys())[k], loc="left",
                        fontsize=12, fontweight="regular", pad=15
                    )
                    axs[i, j+1].imshow(255-self._gv(storage, k)[1], cmap="gray")
                    
                    k += 1
                    j += 1
                            
            if phs_left != 0:  # достраиваем оставшиеся фото в неполной строке
            
                # номер первого подграфика в последней неполной строке или число пустых подграфиков перед первым фото
                first_subplot_number = int((2*self.phs_in_row - phs_left)/2)
                # номер первого пустого после построенных в той же строке
                first_empty_number = first_subplot_number + phs_left
                
                for j in range(first_subplot_number, first_empty_number, 2):
                    axs[lines, j].imshow(self._gv(storage, k)[0], cmap="gray")
                    axs[lines, j].set_title(list(storage.keys())[k], loc='left', fontsize=12, fontweight="regular", pad=15)
                    axs[lines, j+1].imshow(255-self._gv(storage, k)[1], cmap="gray")
                    k += 1
                    j += 1
                    
        elif (lines == 1) & (phs_left == 0):
            for j in range(0, 2*self.phs_in_row, 2):
                axs[j].imshow(self._gv(storage, k)[0], cmap="gray")
                axs[j].set_title(list(storage.keys())[k], loc="left", fontsize=12, fontweight="regular", pad=15)
                axs[j+1].imshow(255-self._gv(storage, k)[1], cmap="gray")
                k += 1
                j += 1
                 
        elif lines == 0:
            for j in range(0, 2*n, 2):
                axs[j].imshow(self._gv(storage, k)[0], cmap="gray")
                axs[j].set_title(list(storage.keys())[k], loc="left", fontsize=12, fontweight="regular", pad=15)
                axs[j+1].imshow(255-self._gv(storage, k)[1], cmap="gray")
                k += 1
                j += 1
       
        for axs in axs.flat:  # убираем оси и рамку на всех подграфиках
            axs.set(xticks=[], yticks=[])
            axs.set_frame_on(False)
                
        plt.show()
        
    def measure(self, storages, dl_names):
        response = {self.model_name: {}}
        
        for storage, dl_name in zip(storages, dl_names):
            response[self.model_name][dl_name] = self.measure_dataloader(storage)
            
        return response
        
    def measure_dataloader(self, storage):
        self.metrics = self._load_metrics()
        
        results = {}
        for metric_name, metric in self.metrics.items():
            values = []
            for img1, img2 in storage.values():
                values.append(metric()(img1, img2))
                                
            values = torch.tensor(values)
            
            raw_tensor_len = len(values)
            values = values[~torch.isnan(values)]
            not_nan_tensor_len = len(values)
            nan_items = raw_tensor_len - not_nan_tensor_len
            
            results[metric_name] = {
                "min": torch.min(values).item(),
                "max": torch.max(values).item(),
                "mean": torch.mean(values).item(),
                "nan_amount": nan_items
            }
            
                
        return results
