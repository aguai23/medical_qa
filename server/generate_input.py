#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import os
data_dir = "/home/yunzhe/Downloads/qa_data_2/data_general/"
question_file = "/home/yunzhe/PycharmProjects/bert/data/question.txt"
answer_file = "/home/yunzhe/PycharmProjects/bert/data/answer.txt"
questions = []
answers = []

for filename in os.listdir(data_dir):
  if filename.endswith(".log") or filename == "LOG_FILE":
    continue
  for record_file in os.listdir(data_dir + filename):
    with open(data_dir + filename + "/" + record_file, "r") as f:
      lines = f.readlines()
      try:
        assert len(lines) == 2
      except AssertionError:
        print(lines[0])
        print(lines[1])
        continue
      # print(lines[0][3:])
      # print(lines[1][3:])
      question = lines[0][3:]
      answer = lines[1][3:]
      questions.append(question)
      answers.append(answer)


print(len(questions))
assert len(questions) == len(answers)


with open(question_file, "w") as question_f:
  with open(answer_file, "w") as answer_f:
    for question in questions:
       question_f.write(question.strip("\n") + "\n")

    for answer in answers:
      answer_f.write(answer.strip("\n") + "\n")
