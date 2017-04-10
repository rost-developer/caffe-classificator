import sys
from argparse import ArgumentParser


class Options:
    """
    this class parse parameters command line
    """
    def __init__(self):
        self._init_parse()

    def _init_parse(self):
        usage = 'bin/task_manager.sh'
        self.parser = ArgumentParser(usage=usage)
        self.parser.add_argument('--train',
                                 type=str,
                                 dest='train_path',
                                 help='An path to train images')

        self.parser.add_argument('--test',
                                 type=str,
                                 dest='test_path',
                                 help='An path to test images')

        self.parser.add_argument('--train_lmdb',
                                 type=str,
                                 dest='train_lmdb',
                                 help='An path to train lmdb')

        self.parser.add_argument('--valid_lmdb',
                                 type=str,
                                 dest='valid_lmdb',
                                 help='An path to validation lmdb')

    def parse(self, args=None):
        return self.parser.parse_args(args)
