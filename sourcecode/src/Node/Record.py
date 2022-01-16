from src.Node.Node import Node

class Record(Node):
    def __init__(self, id, label, children):
        self.id = id
        self.label = label
        super().__init__(children)

