import random
import numpy as np
import torch
from torch.utils.data import Dataset
import argparse

"""
The input-output pairs (x, y) of the NameDataset are of the following form:

  x: Where was Khatchig Mouradian born?⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  x: Where was Jacob Henry Studer born?⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

Using the PAD_CHAR characters in y before the ⁇[place] keeps the trainer from
optimizing the model to predict the question, "Where was...".

Note that the NameDataset should take the pretraining_dataset defined in run.py
as an input. This is to allow the vocab specification of the NameDataset to be
the same as that of the pretraining dataset.
"""
class NameDataset(Dataset):
    def __init__(self, pretraining_dataset, data):
        self.MASK_CHAR = u"\u2047"  # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1"  # the empty square character, for pad
        self.itos = pretraining_dataset.itos
        self.stoi = pretraining_dataset.stoi
        self.block_size = pretraining_dataset.block_size
        self.data = list(data.encode('utf-8').decode('ascii',
                         errors='ignore').split('\n'))

    def __len__(self):
        # returns the length of the dataset
        return len(self.data) - 1

    def __getitem__(self, idx):
        inp, oup = self.data[idx].split('\t')
        x = inp + self.MASK_CHAR + oup + self.MASK_CHAR
        x = x + self.PAD_CHAR*(self.block_size - len(x))
        y = self.PAD_CHAR*(len(inp)-1) + x[len(inp):]

        x = x[:-1]
        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)
        return x, y


"""
Here are some examples of input-output pairs (x, y):

  x: Khatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: hatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

  x: Jaco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: aco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

  x: John Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: ohn Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□


"""
class CharCorruptionDataset(Dataset):
    def __init__(self, data, block_size):
        self.MASK_CHAR = u"\u2047"  # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1"  # the empty square character, for pad

        chars = list(sorted(list(set(data))))
        assert self.MASK_CHAR not in chars
        assert self.PAD_CHAR not in chars
        chars.insert(0, self.MASK_CHAR)
        chars.insert(0, self.PAD_CHAR)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data.split('\n')

    def __len__(self):
        # returns the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # 1. Use the idx argument of getitem to retrieve the element of self.data at the given index. 
        document = self.data[idx]
        
        # 2. Randomly truncate the document to a length no less than 4 characters, and no more than int(self.block_size*7/8) characters. 
        length = random.randint(4, int(self.block_size*7/8))
        if length >= len(document):
            truncated_document = document
        else:
            # start_idx = random.randint(0, len(document) - length)
            # truncated_document = document[start_idx:start_idx + length]
            truncated_document = document[:length]
        
        # 3. Now, break the (truncated) document into three substrings: [prefix] [masked_content] [suffix]
        average_masked_length = len(truncated_document) // 4

        noise = np.random.uniform(-average_masked_length / 8, average_masked_length / 8)
        masked_content_length = average_masked_length + noise
        masked_content_length = int(np.clip(masked_content_length, 1, length - 2))
        start_idx = np.random.randint(1, length - masked_content_length - 1)
        prefix = truncated_document[:start_idx]
        masked_content = truncated_document[start_idx: start_idx + masked_content_length]
        suffix = truncated_document[start_idx + masked_content_length:]
        
        # masked_content_length = random.randint(
        #     average_masked_length // 2, average_masked_length * 2)
        
        # start_idx = random.randint(0, length - masked_content_length)

        # prefix = truncated_document[:start_idx]
        # masked_content = truncated_document[start_idx: start_idx +
        #                                     masked_content_length]
        # suffix = truncated_document[start_idx + masked_content_length:]

        # 4. Rearrange these substrings into the following form: [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] [pads] 
        rearranged_document = prefix + self.MASK_CHAR + \
            suffix + self.MASK_CHAR + masked_content
        rearranged_document += (self.block_size -
                                len(rearranged_document)) * self.PAD_CHAR
        
        assert len(rearranged_document) == self.block_size, f"rearranged_document={len(rearranged_document)}, self.block_size={self.block_size}, they are supposed to be equal\n"
        
        # 5. We now use masked_string to construct the input and output example pair. 
        # To do so, simply take the input string to be masked_string[:-1], 
        # and the output string to be masked_string[1:].
        # In other words, for each character, the goal is to predict the next character in the masked string. 
        ipt, opt = rearranged_document[:-1], rearranged_document[1:]
        
        # 6. Making use of the vocabulary that you defined, encode the resulting input and output strings as Long tensors and return the resulting data point. 
        return torch.LongTensor([self.stoi[c] for c in ipt]), torch.LongTensor([self.stoi[c] for c in opt])


"""
Code under here is strictly for debugging purposes
for example, 
`python src/dataset [namedata|charcorruption]` 
will show the x-y pairs
"""
if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('dataset_type', help="Type of dataset to sample from."
                      "Options: namedata, charcorruption.",
                      choices=["namedata", "charcorruption"])
    args = argp.parse_args()

    if args.dataset_type == 'namedata':
        # Even if it hasn't been implemented, we use it to define the vocab
        corruption_dataset = CharCorruptionDataset(
            open('wiki.txt').read(), 128)
        # Make the name dataset
        name_dataset = NameDataset(corruption_dataset,
                                   open('birth_places_train.tsv').read())
        for _, example in zip(range(4), name_dataset):
            x, y = example
            print('x:', ''.join([name_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([name_dataset.itos[int(c)] for c in y]))
        pass
    elif args.dataset_type == 'charcorruption':
        corruption_dataset = CharCorruptionDataset(
            open('wiki.txt').read(), 128)
        for _, example in zip(range(4), corruption_dataset):
            x, y = example
            print('x:', ''.join([corruption_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([corruption_dataset.itos[int(c)] for c in y]))
    else:
        raise ValueError("Unknown dataset type in command line args: {}"
                         .format(args.dataset_type))
