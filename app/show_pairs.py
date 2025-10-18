import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.visualization import generate_pairs_from_reference, show_positive_pairs, show_negative_pairs

ref_dir = "C:/Users/parth/Downloads/test2"

# Generate positive and negative image pairs
pairs = generate_pairs_from_reference(ref_dir)

# Visualizing pairs
show_positive_pairs(pairs, num_to_show=5)
show_negative_pairs(pairs, num_to_show=5)