from udapi.block.write.textmodetrees import TextModeTrees as Writer
writer = Writer(attributes='form', layout='align', print_comments=True)

def set_fields(fields: str):
    writer.attrs = fields.split()

from udapi.core.root import Root
def write(root: Root):
    writer.process_tree(root)

from typing import Callable
from udapi.core.root import Root
from udapi.core.node import Node

def order_nodes(root: Root, posfunc: Callable[[Node], int]):
    ordering = sorted(root.descendants, key=posfunc)
    for prev, node in zip(ordering[:-1], ordering[1:], strict=True):
        node.shift_after_node(prev, without_children=True)

from itertools import chain
from udapi.core.document import Document
def get_nodes(document: Document) -> list[Node]:
    return list(chain.from_iterable((root.descendants for root in document.trees)))
