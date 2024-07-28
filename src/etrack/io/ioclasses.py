import pathlib
import logging as log
from abc import ABC, abstractmethod

from .dlc_data import DLCReader
from .nixtrack_data import NixtrackData
from ..util import FileType

class DataSource(ABC):

    @abstractmethod
    def data(self):
        log.debug('return data')

    @abstractmethod
    def create_sale_invoice(self, sale):
        log.debug('Creating sale invoice')