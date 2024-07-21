from six.moves.urllib.request import urlopen
import json
import pandas as pd
from argparse import ArgumentParser

### Define command line arguments
parser = ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--pdb_col', type=str)
parser.add_argument('--chain_col', type=str)
parser.add_argument('--no_header', action='store_true')
args = parser.parse_args()

input_file = args.input

if not args.no_header:
    header = True
else:
    header = False

if all([args.pdb_col != None,
        args.chain_col != None]):
    pdb_col = args.pdb_col
    chain_col = args.chain_col

# Load
if '.csv' in input_file:
    csv_df = pd.read_csv(input_file)
    pdb_chain_df = csv_df[[pdb_col, chain_col]]
    pdb_chain_df.columns = ['pdb', 'chain']

else:
    pdb_chain_df = pd.read_csv(input_file, sep='\t', header=None)
    pdb_chain_df.columns = ['pdb', 'chain']

# Open output file for writing
output_file = input_file.replace('.csv', '') + '_uniprot.csv' if '.csv' in input_file else input_file + '_uniprot.tsv'
with open(output_file, 'w') as output:

    # Write header if not specified as no_header
    if not args.no_header:
        output.write(f"{pdb_col},{chain_col},uniprot\n")

    # Process each line
    for pdb, chain in zip(pdb_chain_df['pdb'], pdb_chain_df['chain']):
        print('mapping...', pdb, chain)

        # fetch pdb -> uniprot mapping + check if pdb exists
        try:
            content = urlopen('https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/' + pdb).read()
        except:
            print(pdb, chain, "PDB Not Found (HTTP Error 404).")
            # Write NaN to the output file if PDB ID is not found
            output.write(f"{pdb},{chain},NaN\n")
            continue

        content = json.loads(content.decode('utf-8'))

        # check if chain exists
        chain_exist_boo_list = []

        # find uniprot id
        for uniprot in content[pdb.lower()]['UniProt'].keys():
            for mapping in content[pdb.lower()]['UniProt'][uniprot]['mappings']:
                if mapping['chain_id'] == chain.upper():
                    # Write the result to the output file after processing each line
                    output.write(f"{pdb},{chain},{uniprot}\n")
                    chain_exist_boo_list.append(True)
                    break  # Stop the loop once a matching chain is found
                else:
                    chain_exist_boo_list.append(False)

        if not any(chain_exist_boo_list):
            print(pdb, chain, "PDB Found but Chain Not Found.")
            # Write NaN to the output file if chain ID is not found
            output.write(f"{pdb},{chain},NaN\n")
