import pandas as pd

titanic = pd.read_csv('titanic.csv')

titanic['Name'] = titanic['Name'].str.replace(r'[^\w\s]', '')