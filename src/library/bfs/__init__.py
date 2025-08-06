from queue import Queue
from warnings import warn
from udapi.core.root import Root
from udapi.core.root import Node
from typing import Callable, Optional, TypeVar
from ..graph import Graph, Vertex
from ..ud import order_nodes

T = TypeVar('T')

def make_udapi_tree(graph: Graph[T], summit: T, make_child: Callable[[Node, T], Node], root: Optional[Root] = None) -> Root:
    if root is None:
        root = Root()
    positions = {vertex.content: i for i, vertex in enumerate(graph.vertices)}
    node_positions = dict[Node, int]()
    for vertex in graph.vertices:
        vertex.visited = False
    summit_vertex = graph.dic[summit]
    summit_vertex.visited = True
    summit_node = make_child(root, summit_vertex.content)
    node_positions[summit_node] = positions[summit_vertex.content]
    q = Queue[tuple[T, Node]]()
    q.put((summit_vertex, summit_node))
    while not q.empty():
        vertex, node = q.get()
        for child_vertex in graph.adj[vertex]:
            if not child_vertex.visited:
                child_vertex.visited = True
                child_node = make_child(node, child_vertex.content)
                node_positions[child_node] = positions[child_vertex.content]
                q.put((child_vertex, child_node))
            else:
                print('Visiting for the second time: {0}. Domination ignored: {0} -> {1}.'.format(vertex.content, child_vertex.content))
    try:
        order_nodes(root, node_positions.__getitem__)
    except:
        print(positions)
        print(node_positions)
        raise
    return root
