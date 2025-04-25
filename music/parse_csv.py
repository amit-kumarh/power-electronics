import pandas as pd

df = pd.read_csv("ts2.csv")
print(df.columns)
df['hz'] = 440 * 2**((df['note']-69)/12)

# up 3 octaves
df['hz'] *= 2
df['hz'] *= 2
df['hz'] *= 2

with open("wf_out", 'w') as of:
    of.write("var frequencies = [" + ','.join(df['hz'].astype(str)) + '];\n')
    of.write("var timePoints = [" + ','.join(df['ts'].astype(str)) + '];\n')

print(min(df['hz']), max(df['hz']))
