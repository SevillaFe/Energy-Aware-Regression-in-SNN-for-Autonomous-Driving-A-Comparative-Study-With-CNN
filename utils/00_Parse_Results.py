import os
import re
import pandas as pd

import utils.utils as myUtils


if __name__ == "__main__":
    root_dir = '/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images'
    
    
    df = myUtils.load_and_parse_results(root_dir)
    df.to_csv(os.path.join(root_dir, 'Tensorflow_parsed_results.csv'), index=False)
    print("Parsed results saved to Tensorflow_parsed_results.csv")