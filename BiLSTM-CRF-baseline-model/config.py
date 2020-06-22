from pathlib import Path

data_dir = Path("./dataset/chem/")
train_path = data_dir / 'train.txt'
dev_path =data_dir / 'dev.txt'
test_path = data_dir / 'test.txt'
output_dir = Path("./outputs")

label2id = {
    "O": 0,
    "B":1,
    "E":2,
    "M":3,
    'S':4,
    "<START>": 5,
    "<STOP>": 6
}
