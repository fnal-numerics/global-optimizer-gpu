import pandas as pd

# Load the TSV file
df = pd.read_csv('zeus_100d_results.tsv', sep='\t')

# Extract and rename columns: run→Iter, N→NumOpt
table = df[['run', 'N', 'time', 'fval', 'error']].copy()
table.columns = ['Iter', 'NumOpt', 'Time (s)', 'Fval', 'Error']

# Format numerics for nicer LaTeX
table['Time (s)'] = table['Time (s)'].map('{:.2f}'.format)
table['Fval']    = table['Fval'].map('{:.2f}'.format)
table['Error']   = table['Error'].map('{:.2e}'.format)

# Build the LaTeX code
lines = []
lines.append(r'\begin{table}[ht]')
lines.append(r'\centering')
lines.append(r'\begin{tabular}{ccccc}')
lines.append(r'\hline')
lines.append(' & '.join(table.columns) + r' \\')
lines.append(r'\hline')
for _, row in table.iterrows():
    lines.append(' & '.join(row.astype(str)) + r' \\')
lines.append(r'\hline')
lines.append(r'\end{tabular}')
lines.append(r'\caption{Summary of Rosenbrock optimization results with error column.}')
lines.append(r'\label{tab:rosenbrock_error}')
lines.append(r'\end{table}')

# Print the LaTeX table
print('\n'.join(lines))

