import pandas as pd
import os

def get_store_item(df,s100,i100,c100,c101):
    return df[(df['S100']==s100) & (df['I100']==i100)& (df['C100']==c100)& (df['C101']==c101) ]