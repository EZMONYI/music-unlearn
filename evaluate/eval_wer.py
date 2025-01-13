import argparse

parser = argparse.ArgumentParser(
    description="evaluate wer")

parser.add_argument("--infer", "-i", type=str, help="inference output")
parser.add_argument("--refer", "-r", type=str, help="reference file")


def calculate_wer(reference, hypothesis):
    """
    Calculate the Word Error Rate (WER) between two lines of text.

    Parameters:
        reference (str): The ground truth text.
        hypothesis (str): The predicted text.

    Returns:
        float: The WER as a percentage.
    """
    # Tokenize the input lines
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Initialize the matrix
    len_ref = len(ref_words)
    len_hyp = len(hyp_words)
    dp = [[0 for _ in range(len_hyp + 1)] for _ in range(len_ref + 1)]

    # Fill the base cases
    for i in range(len_ref + 1):
        dp[i][0] = i  # Deletions
    for j in range(len_hyp + 1):
        dp[0][j] = j  # Insertions

    # Populate the matrix
    for i in range(1, len_ref + 1):
        for j in range(1, len_hyp + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No error
            else:
                substitution = dp[i - 1][j - 1] + 1
                insertion = dp[i][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                dp[i][j] = min(substitution, insertion, deletion)

    # Calculate WER
    wer = dp[len_ref][len_hyp] / len_ref
    return wer

def main(args):
    refs = {}
    hypos = {}
    with open(args.infer, "r") as f:
        for line in f:
            id, _, tokens = line.split("\t")
            id = id.split("-")[1]
            if id not in hypos.keys():
                hypos[id] = tokens
            else:
                continue
    with open(args.refer, "r") as f:
        for line in f:
            id, tokens = line.split("\t")
            refs[id.split("-")[1]] = tokens


    total_num = 0
    wer = 0
    for k, v in hypos.items():
        wer = wer + calculate_wer(v, refs[k])
        total_num = total_num + 1

    print(wer / total_num)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
