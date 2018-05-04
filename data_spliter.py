import random
import math

"""
data_splitter() splits the lines of input files into multiple output files, distributing each class uniformly.
in_file_name: the address of the file to read in
out_file_names: a list of file locations to output to
fractions: a list of floats to specify what fraction of the lines should go to each file.
return: 0 on success, 1 on detected error
"""


def data_spliter(in_file_name, out_file_names, fractions):
    """
    :param in_file_name: CSV file to be split.
    :param out_file_names: CSV files to save split dataset.
    :param fractions: The fraction of dataset each out file will receive.
    :return: 1 if fractions don't sum to 1, or 0 otherwise.
    """
    if not math.isclose(sum(fractions), 1, rel_tol=1e-5):
        return 1
    num_in_lines = 0
    class_counts = {}
    with open(in_file_name, 'r') as in_file:
        for line in in_file:
            num_in_lines += 1
            line_class = line.split(',')[0]
            if not line_class in class_counts:
                 class_counts[line_class] = 0
            class_counts[line_class] += 1
        in_file.seek(0)
        out_files = []
        num_writes_left = []
        for frac in fractions:
            out_file_class_counts = {}
            for class_name in class_counts:
                out_file_class_counts[class_name] = math.ceil(frac * class_counts[class_name])
            num_writes_left.append(out_file_class_counts)
        for out_file_name in out_file_names:
            out_files.append(open(out_file_name, "w"))
        for line in in_file:
            num = random.random()   # generates a pseudo-random number between 0 and 1
            file_classifier_bound = 0
            line_class = line.split(',')[0]
            for i in range(len(out_files)):
                file_classifier_bound += fractions[i]
                if num <= file_classifier_bound:
                    if num_writes_left[i][line_class] != 0:
                        num_writes_left[i][line_class] -= 1
                        out_files[i].write(line)
                    else:
                        for j in range(len(out_files)):
                            if num_writes_left[j][line_class] != 0:
                                num_writes_left[j][line_class] -= 1
                                out_files[j].write(line)
                    break

        for out_file in out_files:
            out_file.close()
        in_file.close()
        return 0