from abc import ABC, abstractmethod, abstractproperty
from typing import List, Union

from .rs_captions import RsMetadataIndex    # type: ignore
from .index import Documents


class MetadataFormat(ABC):

    @staticmethod
    def header(doc_id: int, n: int) -> bytes:
        return doc_id.to_bytes(4, 'little') + n.to_bytes(4, 'little')

    @abstractmethod
    def decode(self, s: bytes) -> object:
        """Return decoded metadata"""
        pass

    @abstractproperty
    def size(self) -> int:
        """Number of bytes of metadata"""
        pass


class MetadataIndex(object):
    """
    Interface to binary encoded metadata files for efficient iteration
    """

    DocIdOrDocument = Union[int, Documents.Document]

    def __init__(self, path: str, documents: Documents,
                 metadata_format: MetadataFormat, debug: bool = False):
        assert isinstance(metadata_format, MetadataFormat)
        assert metadata_format.size > 0, \
            'Invalid metadata size: {}'.format(metadata_format.size)
        self._documents = documents
        self._meta_fmt = metadata_format
        self._rs_meta = RsMetadataIndex(path, metadata_format.size, debug)

    def __require_open_index(f):
        def wrapper(self, *args, **kwargs):
            if self._rs_meta is None:
                raise ValueError('I/O on closed MetadataIndex')
            return f(self, *args, **kwargs)
        return wrapper

    @__require_open_index
    def metadata(
        self, doc: 'MetadataIndex.DocIdOrDocument',
        position: int = 0, count: int = 2 ** 31
    ) -> List[object]:
        """
        Generator over metadata returned by the MetadataFormat's decode method.
        """
        doc_id = self.__get_document_id(doc)
        if position < 0:
            raise ValueError('Position cannot be negative')
        if count < 0:
            raise ValueError('Count cannot be negative')
        return [
            self._meta_fmt.decode(b)
            for b in self._rs_meta.metadata(doc_id, position, count)]

    def close(self) -> None:
        self._rs_meta = None

    def __enter__(self) -> 'MetadataIndex':
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        self.close()

    def __get_document_id(self, doc: 'MetadataIndex.DocIdOrDocument') -> int:
        if isinstance(doc, Documents.Document):
            return doc.id
        else:
            return self._documents[doc].id
