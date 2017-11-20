Usage:

python3 train.py <userid> <folder where models are saved>
Path of the folder is files/models/<what you name it>

Input format:

Each caption (with start and end tokens) is converted into a one-hot matrix of shape MAX_SEQUENCE_LENGTH x VOCAB_SIZE.
Output is this matrix rotated one row upwards, so first word has second word corresponding to it.