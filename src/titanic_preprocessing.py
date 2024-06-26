import pandas as pd

titanic = pd.read_csv('./datasets/titanic.csv')

# titanic['Title'] = titanic['Name'].str.extract(r'^(\w+)', expand=False)

# coded = pd.get_dummies(titanic['Title'], prefix='title', dtype=int)

titanic['Male'] = titanic['Sex'].map({'male': 1, 'female': 0})

# titanic = titanic.drop(columns={'Name','Title','Sex'})
titanic = titanic.drop(columns={'Sex', 'Name'})

# titanic = pd.concat([titanic, coded], axis=1)

titanic = titanic.rename(columns={'Survived': 'class'})

titanic.to_csv('./datasets/titanic_processed.csv', index=False)