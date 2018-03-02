import random
import math


def data_spliter(in_file_name, out_file_names, fractions):
    if not math.isclose(sum(fractions), 1, rel_tol = 1e-5):
        return 1
    num_in_lines = 0
    with open(in_file_name, 'r') as in_file:
        for _ in in_file:
            num_in_lines += 1
        in_file.seek(0)
        out_files = []
        num_writes_left = []
        for frac in fractions:
            num_writes_left.append(math.ceil(frac * num_in_lines))
        for out_file_name in out_file_names:
            out_files.append(open(out_file_name, "w"))
        for line in in_file:
            num = random.random()   # generates a pseudo-random number between 0 and 1
            file_classifier_bound = 0
            for i in range(len(out_files)):
                file_classifier_bound += fractions[i]
                if num <= file_classifier_bound:
                    if num_writes_left[i] != 0:
                        num_writes_left[i] -= 1
                        out_files[i].write(line)
                    else:
                        for j in range(len(out_files)):
                            if num_writes_left[j] != 0:
                                num_writes_left[j] -= 1
                                out_files[j].write(line)
                    break


        for out_file in out_files:
            out_file.close()
        in_file.close()
        return 0