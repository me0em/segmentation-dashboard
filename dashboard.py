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
from json2html import *
import tempfile
import webbrowser
import os
import json
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
    """ Dashboard class
    
    Args:
        dataloaders <list[dataloader]>: list of dataloader objects which are returns tuple of any types objects.
                                        every dataloader should have name attribute for JSON report
        model: model returns object that go right in the measure() and go in the render() wrapped with render_fn
        model_name: name of the model for JSON report
        render_fn: use this fn to wraps dataloader and modelled object to render
        render_dl_idx <int>: index of dataloader we need to render
        obj_item <int>: index of element in tuple from dataloader that will be rendered
        obj_name_item <int>: index of filename
        to_model_item <int>: image
        limit <int>: amount of rendered objects
        phs_in_row <int>: amount of rendered objects in row
        get_randomly <bool>: it's usefull if dataloader is big and we want to get different
                             :limit: object from it to render  
        c <int>: frankly idk what it is
        inverse_modelled_obj <bool>: if True modelled tensor [1, 0, 0] flips to [0, 1, 1]
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
            c=3,
            inverse_modelled_obj=False
        ):
        
        # model vars
        self.dataloaders = dataloaders
        self.model = model.eval()  # EVAL IT
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
        self.inverse_modelled_obj = inverse_modelled_obj

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

                filepath = data[self.obj_name_item][0]  # IF BATCH_SIZE=1 WE NEED [0]
                target_obj = data[self.obj_item]
                modelled_obj = self.model.predict(data[self.to_model_item])
                if self.inverse_modelled_obj:
                    modelled_obj = torch.abs(1 - modelled_obj)
                    
                storage[filepath] = (target_obj.cpu(), modelled_obj.cpu())  # save to RAM as cpu
                
            storages.append(storage)
            dl_names.append(dl.dataset.name)
        
        # return storages
        
        logger.info("Storages has been formed")
                
        logger.info("Render...") 
        self.render(storages[self.render_dl_idx])
        
        logger.info("Measure...")
        report = self.measure(storages, dl_names)
        
        return report
        
        logger.info("Dump results... / TODO")
        # self.save_report(storage)

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
                    axs[i, j+1].imshow(self._gv(storage, k)[1], cmap="gray")
                    
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
                    axs[lines, j+1].imshow(self._gv(storage, k)[1], cmap="gray")
                    k += 1
                    j += 1
                    
        elif (lines == 1) & (phs_left == 0):
            for j in range(0, 2*self.phs_in_row, 2):
                axs[j].imshow(self._gv(storage, k)[0], cmap="gray")
                axs[j].set_title(list(storage.keys())[k], loc="left", fontsize=12, fontweight="regular", pad=15)
                axs[j+1].imshow(self._gv(storage, k)[1], cmap="gray")
                k += 1
                j += 1
                 
        elif lines == 0:
            for j in range(0, 2*n, 2):
                axs[j].imshow(self._gv(storage, k)[0], cmap="gray")
                axs[j].set_title(list(storage.keys())[k], loc="left", fontsize=12, fontweight="regular", pad=15)
                axs[j+1].imshow(self._gv(storage, k)[1], cmap="gray")
                k += 1
                j += 1
       
        for axs in axs.flat:  # убираем оси и рамку на всех подграфиках
            axs.set(xticks=[], yticks=[])
            axs.set_frame_on(False)
                
        plt.show()
        
    def measure(self, storages, dl_names):
        report = {self.model_name: {}}
        
        for storage, dl_name in zip(storages, dl_names):
            report[self.model_name][dl_name] = self.measure_dataloader(storage)
            
        return report
        
    def measure_dataloader(self, storage, device="cuda:0"):
        self.metrics = self._load_metrics()
        
        dataloader_report = {}
        for metric_name, metric in self.metrics.items():
            values = []
            for img1, img2 in storage.values():
                values.append(metric().to(device)(img1.to(device), img2.to(device)))
                                
            values = torch.tensor(values)
            
            raw_tensor_len = len(values)
            values = values[~torch.isnan(values)]
            not_nan_tensor_len = len(values)
            nan_items = raw_tensor_len - not_nan_tensor_len
            
            dataloader_report[metric_name] = {
                "min": torch.min(values).item(),
                "max": torch.max(values).item(),
                "mean": torch.mean(values).item(),
                # "nan_amount": nan_items
            }
                
        return dataloader_report

    @staticmethod
    def view(report):
        html = json2html.convert(json=report, table_attributes='border="1px"; border-color: rgba(0,0,255,0.25)')
        
        html = f'<link rel="stylesheet" href="{os.getcwd()}/table.css">' + \
         '<h1><center>Dashboard 📊</center></h1>' + \
         '<div class="table-wrapper"><table class="fl-table">' + \
         html + \
         '</table></div>'

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as tf:
            tf.write(html)
            webbrowser.open('file://' + os.path.realpath(tf.name))
            
    @staticmethod
    def dump(report):
        while True:
            inpt = input("Do you want add report in the storage? ('y' or 'n')")
            
            if inpt not in ("y", "yes", "n", "no"):
                print("'y' if yes; 'n' if no")
    
            elif inpt in ("n", "no"):
                return
            
            elif inpt in ("y", "yes"):
                with open("storage.json", "r") as json_file:
                    json_decoded = json.load(json_file)

                json_decoded[list(report.keys())[0]] = list(report.values())[0]

                with open("storage.json", 'w') as json_file:
                    json.dump(json_decoded, json_file)
                    
                return