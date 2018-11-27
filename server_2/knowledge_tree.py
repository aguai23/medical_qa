import os


class KnowledgeNode(object):

  def __init__(self, value, parent, connection):
    """
    construct one entity node in knowledge tree
    :param value: str, node value
    :param parent: parent node
    :param connection: str, connection relation description, could be none
    """
    self.value = value
    self.parent = parent
    self.connection = connection
    self.children = []

  def add_child(self, child):
    self.children.append(child)

  def is_root(self):
    return self.parent is None


class KnowledgeTree(object):

  def __init__(self, data_path):
    self.root = None
    self.build_tree(data_path)

  def build_tree(self, data_path):
    with open(data_path, "r") as f:
      lines = f.readlines()
      triples = []
      for line in lines:
        triple = line.replace("\n","").split("|")
        triples.append(triple)
        if len(triple) == 2:
          parent, child = triple
          connection = ""
        else:
          parent, connection, child = triple
        child_node = KnowledgeNode(child, None, connection)
        if self.root is None:
          self.root = KnowledgeNode(parent, None, None)
        parent_node = self.search_parent(parent)
        child_node.parent = parent_node
        parent_node.add_child(child_node)

  def search_parent(self, parent_value):
    """
    search the node with exact name, use bfs search
    :param parent_value: the value of corresponding node
    :return: target node
    """
    queue = [self.root]
    while len(queue) > 0:
      node = queue.pop(0)
      if node.value == parent_value:
        return node
      queue += node.children

    return None

  def print_tree(self):
    queue = [self.root]
    next_layer = []
    while len(queue) > 0:
      node = queue.pop(0)
      next_layer += node.children
      # if not node.value.startswith("***"):
      #   next_layer.append(KnowledgeNode("***" + node.value + "***", None, None))
      if len(queue) == 0:
        print(node.value)
        queue += next_layer
        next_layer = []
      else:
        print(node.value, end="|")


if __name__ == "__main__":
  knowledge_tree = KnowledgeTree("./data/graph")
  knowledge_tree.print_tree()