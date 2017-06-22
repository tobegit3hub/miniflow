import ops


class Graph(object):
  def __init__(self):
    self.name_op_map = {}

  def get_unique_name(self, original_name):
    index = 0
    unique_name = "{}_{}".format(original_name, index)

    while unique_name in self.name_op_map.keys():
      index += 1
      unique_name = "{}_{}".format(original_name, index)

    return unique_name

  def add_to_graph(self, op):
    op.name = self.get_unique_name(op.name)
    self.name_op_map[op.name] = op


# TODO: Make global variable for all packages
default_graph = Graph()


def get_default_graph():
  if default_graph == None:
    global default_graph
    default_graph = Graph()
  else:
    return default_graph


class Session(object):
  def __init__(self):
    pass

  def run(self, op, feed_dict=None, options=None):

    # Update the value of PlaceholerOp with feed_dict data
    name_op_map = op.graph.name_op_map

    if feed_dict != None:
      # Example: {"Placeholer_1": 10} or {PlaceholderOp: 10}
      for op_or_opname, value in feed_dict.items():
        if isinstance(op_or_opname, str):
          placeholder_op = name_op_map[op_or_opname]
        else:
          placeholder_op = op_or_opname
        if isinstance(placeholder_op, ops.PlaceholderOp):
          placeholder_op.set_value(value)

    result = op.forward()
    return result
