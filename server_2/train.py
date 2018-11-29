import tensorflow as tf
from data_processor import DataProcessor
from knowledge_tree import KnowledgeTree
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
  "graph_path", None,
  "The graph path file, containing all the triples constructing knowledge graph"
)

flags.DEFINE_string(
  "data_path", None,
  "the labeled questions path"
)

flags.DEFINE_string(
  "output_dir", None,
  "the output directory where the model checkpints will be written"
)


def main(_):
  knowledge_tree = KnowledgeTree(FLAGS.graph_path)
  data_processor = DataProcessor(FLAGS.data_path, knowledge_tree)
  training_sample, training_label = data_processor.get_training_samples()


if __name__ == "__main__":
  flags.mark_flag_as_required("graph_path")
  flags.mark_flag_as_required("data_path")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()