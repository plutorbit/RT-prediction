import pandas as pd 
import numpy as np 
from Bio.SeqUtils.ProtParam import ProteinAnalysis

df = pd.read_csv('/Users/sarahackerman/Desktop/grouped_cystein5.csv')


def physchem_prop (sequence): 
    analyzed_seq = ProteinAnalysis(sequence)
    return{
        'molecular_weight':  analyzed_seq.molecular_weight(),
        'instability_index': analyzed_seq.instability_index(),
        'isoelectric_point': analyzed_seq.isoelectric_point()
    }
def seq_length (sequence):
    return len(sequence)

def extracting (sequence): 
    psychchem_seq = physchem_prop(sequence)
    length = seq_length(sequence)
    
    features = {'seq_length': length, **psychchem_seq} # '**' used for merging dictionaries 
    return features     


new_df = df['sequence'].apply(extracting)


new_df = pd.json_normalize(new_df) #normalizing values in new columns 
df_features = pd.concat([df.reset_index(drop=True), new_df.reset_index(drop=True)], axis=1)

output = '/Users/sarahackerman/Desktop/grouped_cystein5_feature_ex.csv'
df_features.to_csv(output,index=False)

print('saved')

