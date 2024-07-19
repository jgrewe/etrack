import pathlib
import logging as log
from abc import ABC, abstractmethod

from .dlc_data import DLCReader
from .nixtrack_data import NixtrackData


class DataSource(ABC):

    def open(self, filename, cr):
        p = pathlib.Path(filename)
        log.debug(f"Open file {filename}")
        if p.suffix == ".h5":
            self._data = DLCReader(filename)
        pass

    @abstractmethod
    def data(self):
        return self._data

    @abstractmethod
    def create_sale_invoice(self, sale):
        log.debug('Creating sale invoice')