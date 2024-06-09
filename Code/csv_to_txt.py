import pandas as pd

name = 'G_news'
df = pd.read_csv('./data_stego/7b/'+name+'.csv')
selected_column = df['stega_bits']

selected_column.to_csv('./data_stego/7b/'+name+'.bit', index=False, header=False)
