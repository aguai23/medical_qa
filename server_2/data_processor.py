import jieba
from zhon.hanzi import punctuation
import re
import gensim
import numpy as np


class DataProcessor(object):

  def __init__(self, data_file, knowledge_tree):
    self.training_data = []
    self.train_label = []
    self.valid_data = []
    self.valid_label = []
    self.questions, self.answers = self.load_data(data_file)
    self.build_samples()
    self.knowledge_tree = knowledge_tree
    self.add_positive_samples(self.knowledge_tree)
    self.add_negative_samples(self.knowledge_tree)

  def load_data(self, data_file):
    questions = []
    answers = []
    with open(data_file, "r") as f:
      lines = f.readlines()
      question = True
      for line in lines:
        if question:
          questions.append(line.replace("\n", ""))
        else:
          parents, children = self.parse_answer(line)
          answers.append((parents, children))
        question = not question
    return questions, answers

  def build_samples(self):
    # represent current description to be output
    description_pairs = set()
    # represent process it's child node
    continue_pairs = set()
    for i in range(len(self.questions)):
      question = self.questions[i]
      parents, children = self.answers[i]
      for j in range(len(parents)):
        parent = parents[j]
        child = children[j]
        description_pairs.add((question, child))
        continue_pairs.add((question, parent))

    # for training label, the first indicate output current description
    # the second indicate continue to child node
    for description_pair in description_pairs:
      self.training_data.append(description_pair)
      if description_pair in continue_pairs:
        self.train_label.append((1, 1))
        continue_pairs.remove(description_pair)
      else:
        self.train_label.append((1, 0))

    for continue_pair in continue_pairs:
      self.training_data.append(continue_pair)
      self.train_label.append((0, 1))

  def add_positive_samples(self, knowledge_tree):
    """
    we only have the examples exist in question label data, which means
    hidden relations are ignored, we have to add them back
    :param knowledge_tree: the knowledge tree we have built
    :return:
    """
    print("original training number " + str(len(self.training_data)))
    supplement_samples = []
    supplement_labels = []
    training_set = set(self.training_data)
    for positive_sample in self.training_data:
      question = positive_sample[0]
      entity_name = positive_sample[1]
      node = knowledge_tree.search_node(entity_name)
      while node.parent is not None:
        parent_name = node.parent.value
        if (question, parent_name) not in training_set:
          training_set.add((question, parent_name))
          supplement_samples.append((question, parent_name))
          supplement_labels.append((0, 1))
        node = node.parent
    self.training_data += supplement_samples
    self.train_label += supplement_labels
    print("training number with all positive examples " + str(len(self.training_data)))

  def add_negative_samples(self, knowledge_tree):
    """
    add all negative samples the question will encounter when pass from root node
    :param knowledge_tree:
    :return:
    """
    negative_samples = []
    negative_labels = []
    train_set = set(self.training_data)
    for positive_sample in self.training_data:
      question = positive_sample[0]
      entity_name = positive_sample[1]
      node = knowledge_tree.search_node(entity_name)
      while node.parent is not None:
        siblings = node.parent.children
        for child in siblings:
          if (question, child.value) not in train_set:
            train_set.add((question, child.value))
            negative_samples.append((question, child.value))
            negative_labels.append((0, 0))
        node = node.parent
    self.training_data += negative_samples
    self.train_label += negative_labels
    print("training number with negative examples " + str(len(self.training_data)))

  def get_training_samples(self):

    assert len(set(self.training_data)) == len(self.training_data)
    model = gensim.models.Word2Vec.load("./data/word2vec")
    # segment words and convert to word embedding representations
    vector_train_data = []
    default_embedding = np.zeros(100)
    for sample in self.training_data:
      question = sample[0]
      entity = sample[1]
      question = re.sub(u"[%s]+" % punctuation, "", question)
      question_tokens = jieba.cut(question)
      entity_tokens = jieba.cut(entity)
      question_feature = []
      for token in question_tokens:
        if token not in model.wv:
          question_feature.append(default_embedding)
        else:
          embedding = model.wv[token]
          question_feature.append(embedding)
      entity_feature = []
      for token in entity_tokens:
        if token not in model.wv:
          entity_feature.append(default_embedding)
        else:
          embedding = model.wv[token]
          entity_feature.append(embedding)
      question_feature = np.asarray(question_feature)
      entity_feature = np.asarray(entity_feature)
      # print(question_feature.shape)
      # print(entity_feature.shape)
      vector_train_data.append((question_feature, entity_feature))
    self.training_data = vector_train_data
    # print(len(self.training_data))
    # print(len(self.train_label))
    return self.training_data, self.train_label

  @staticmethod
  def parse_answer(answer):
    triples = answer.replace("\n", "").split("||")
    assert len(triples) > 0
    parents = []
    children = []
    for triple in triples:
      entities = triple.split("|")
      assert len(entities) == 3 or len(entities) == 2
      parents.append(entities[0])
      if len(entities) == 3:
        children.append(entities[2])
      else:
        children.append(entities[1])
    return parents, children


if __name__ == "__main__":
  data_processor = DataProcessor("./data/question_label")
