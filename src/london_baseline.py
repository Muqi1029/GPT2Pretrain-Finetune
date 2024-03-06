# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
from utils import evaluate_places

DEV_SET = "birth_dev.tsv"

length = len(list(open(DEV_SET)))
total, correct =evaluate_places(DEV_SET, ['London'] * length)
print(f"Correct: {correct} out of {total}: {correct / total * 100}%")
