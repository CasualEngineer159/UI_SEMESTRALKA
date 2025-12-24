import pandas as pd

data = [
    "jablko",
    "banán",
    "hruška",
    "jablko"
]
df = pd.DataFrame(data, columns=["ovoce"])

df_legend = {}

def encode_column(column):
    legend = {}
    def encode(value):
        if value not in legend:
            legend[value] = len(legend)
        return legend[value]

    column = column.apply(encode)
    return column, legend

df["ovoce"], df_legend["ovoce"] = encode_column(df["ovoce"])
print(df)
print(df_legend)
