from typing import Generic, TypeVar
T = TypeVar('T')

class Vertex(Generic[T]):

    def __init__(self, content: T):
        self.content = content
        self.visited = False

    def reset(self):
        self.visited = False

    def __repr__(self):
        return 'Vertex(' + self.content.__repr__() + ')'

class Graph(Generic[T]):

    def __init__(self, contents: list[T], edges: list[tuple[T, T]]):
        self.vertices = list[Vertex[T]]()
        self.dic = dict[T, Vertex]()
        for content in contents:
            vertex = Vertex(content)
            self.vertices.append(vertex)
            self.dic[content] = vertex
        self.adj = dict[Vertex[T], list[Vertex[T]]]()
        for vertex in self.vertices:
            self.adj[vertex] = list[Vertex[T]]()
        for source, target in edges:
            self.adj[self.dic[source]].append(self.dic[target])

    def __tuple__(self):
        return self.vertices, self.adj

    def __repr__(self):
        return self.__tuple__().__repr__()
