import math
import pandas as pd


# positive = more hydrophobic
KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5,
    'C': 2.5, 'E': -3.5, 'Q': -3.5, 'G': -0.4,
    'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
    'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
    'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}


PKA_DICT = {
    'K': 10.5,
    'R': 12.5,
    'H': 6.0,
    'D': 3.9,
    'E': 4.2,
    'C': 8.3,
    'Y': 10.1
}

# pKa for terminal groups
PKA_N_TERMINUS = 9.0
PKA_C_TERMINUS = 2.0


def compute_net_charge(sequence, pH=7.0):
    """
    Compute the net charge of a peptide at a given pH using
    a simplified Henderson–Hasselbalch approach.
    """
    # 2.1 Charge from N-terminus
    n_term_charge = 1.0 / (1.0 + 10 ** (pH - PKA_N_TERMINUS))

    # 2.2 Charge from C-terminus
    c_term_charge = -1.0 / (1.0 + 10 ** (PKA_C_TERMINUS - pH))

    total_charge = n_term_charge + c_term_charge

    for aa in sequence:
        if aa in PKA_DICT:
            ratio = 1.0 / (1.0 + 10 ** (pH - PKA_DICT[aa]))

            if aa in ['K', 'R', 'H']:
                # Basic residues: fraction protonated => +1 * ratio
                residue_charge = ratio
            else:
                # Acidic residues: fraction deprotonated => -1 * (1 - ratio)
                residue_charge = -(1.0 - ratio)

            total_charge += residue_charge

    return total_charge


def compute_gravy(sequence):
    """
    Compute the Grand Average of Hydropathy (GRAVY) using
    the Kyte & Doolittle scale.
    """
    if len(sequence) == 0:
        return 0.0

    total_hydro = 0.0
    for aa in sequence:
        total_hydro += KYTE_DOOLITTLE.get(aa, 0.0)  # default 0 if unknown

    return total_hydro / len(sequence)


def compute_aliphatic_index(sequence):
    """
    Compute Aliphatic Index (AI). One common formula is:
      AI = 100 * (nA + 2.9 * nV + 3.9 * (nI + nL)) / total_length
    """
    length = len(sequence)
    if length == 0:
        return 0.0

    nA = sequence.count('A')
    nV = sequence.count('V')
    nI = sequence.count('I')
    nL = sequence.count('L')

    ai = (100.0 * (nA + 2.9 * nV + 3.9 * (nI + nL))) / length
    return ai


def compute_hydrophobicity(sequence):
    """
    Compute the count and fraction of hydrophobic residues.
    Define 'hydrophobic' as: I, L, V, F, W, M, A, C (example set).
    """
    hydrophobic_set = {'I', 'L', 'V', 'F', 'W', 'M', 'A', 'C'}

    count_hydro = sum(aa in hydrophobic_set for aa in sequence)
    length = len(sequence)
    frac_hydro = count_hydro / length if length > 0 else 0.0

    return count_hydro, frac_hydro


def compute_aromaticity(sequence):
    """
    Compute the count and fraction of aromatic residues.
    Usually, F, W, Y are considered aromatic (sometimes H).
    """
    aromatic_set = {'F', 'W', 'Y'}
    length = len(sequence)

    count_arom = sum(aa in aromatic_set for aa in sequence)
    frac_arom = count_arom / length if length > 0 else 0.0

    return count_arom, frac_arom


def compute_grouped_fractions(sequence):
    """
    Break down fraction of:
    1) (I, L, V, F, W)
    2) (D, E, K, R)
    3) (S, T, Y)
    """
    group1 = {'I', 'L', 'V', 'F', 'W'}
    group2 = {'D', 'E', 'K', 'R'}
    group3 = {'S', 'T', 'Y'}

    length = len(sequence)
    if length == 0:
        return 0.0, 0.0, 0.0

    count_g1 = sum(aa in group1 for aa in sequence)
    count_g2 = sum(aa in group2 for aa in sequence)
    count_g3 = sum(aa in group3 for aa in sequence)

    frac_g1 = count_g1 / length
    frac_g2 = count_g2 / length
    frac_g3 = count_g3 / length

    return frac_g1, frac_g2, frac_g3


def compute_peptide_features(sequence, pH=7.0):

    seq_length = len(sequence)

    # Basic features
    gravy = compute_gravy(sequence)
    aliphatic_idx = compute_aliphatic_index(sequence)
    net_charge = compute_net_charge(sequence, pH=pH)

    # Hydrophobicity & Aromaticity
    count_hydro, frac_hydro = compute_hydrophobicity(sequence)
    count_arom, frac_arom = compute_aromaticity(sequence)

    # Group fractions
    frac_g1, frac_g2, frac_g3 = compute_grouped_fractions(sequence)

    # Cysteine count
    cys_count = sequence.count('C')

    # Derived features
    cys_x_hydro = cys_count * gravy  # "Cysteine count × GRAVY"
    length_x_arom = seq_length * frac_arom  # "Sequence length × Aromaticity"

    features = {
        "sequence_length": seq_length,
        "gravy": gravy,
        "aliphatic_index": aliphatic_idx,
        "net_charge_at_pH_{}".format(pH): net_charge,

        "count_hydrophobic": count_hydro,
        "fraction_hydrophobic": frac_hydro,
        "count_aromatic": count_arom,
        "fraction_aromatic": frac_arom,

        "fraction_ILVFW": frac_g1,  # (I, L, V, F, W)
        "fraction_DEKR": frac_g2,  # (D, E, K, R)
        "fraction_STY": frac_g3,  # (S, T, Y)

        "cysteine_count": cys_count,
        "cysteine_count_x_GRAVY": cys_x_hydro,
        "sequence_length_x_aromaticity": length_x_arom
    }

    return features


def add_peptide_features_to_csv(input_csv, output_csv, sequence_col="sequence", pH=7.0):

    df = pd.read_csv(input_csv)


    all_new_features = {}

    # Initialize the dictionary keys based on what compute_peptide_features returns
    example_features = compute_peptide_features("TEST", pH=pH)
    for feat_key in example_features.keys():
        all_new_features[feat_key] = []

    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        seq = row[sequence_col]

        # Ensure sequence is a string
        if not isinstance(seq, str):
            seq = str(seq)

        # Compute features
        feats = compute_peptide_features(seq, pH=pH)

        # Append each feature to the corresponding list
        for feat_key, feat_value in feats.items():
            all_new_features[feat_key].append(feat_value)

    # Create new columns in the DataFrame
    for feat_key, feat_values in all_new_features.items():
        df[feat_key] = feat_values

    # Write to a new CSV
    df.to_csv(output_csv, index=False)
    print(f"New CSV with features written to: {output_csv}")


if __name__ == "__main__":

    input_file = "dataset_paper.csv"
    output_file = "peptides_with_features.csv"

    add_peptide_features_to_csv(
        input_csv=input_file,
        output_csv=output_file,
        sequence_col="sequence",
        pH=7.0
    )
