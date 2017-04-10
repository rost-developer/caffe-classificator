import os
import glob

import sys

from lib.img_preparing import ImgPreparing
from lib.options import Options

if __name__ == '__main__':
    options = Options()
    opts = options.parse(sys.argv[1:])
    img_prepare = ImgPreparing()

    os.system('rm -rf  ' + opts.train_lmdb)
    os.system('rm -rf  ' + opts.validation_lmdb)

    # Get information about images
    train_data = [img for img in glob.glob(opts.train)]
    test_data = [img for img in glob.glob(opts.test)]

    # Create binary views for files
    img_prepare.create_lmdb(train_data, opts.train_lmdb)
    img_prepare.create_lmdb(train_data, opts.validation_lmdb)

