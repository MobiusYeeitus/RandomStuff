import pandas as pd

from pandas import read_csv

iris=read_csv('https://raw.githubusercontent.com/MobiusYeeitus/RandomStuff/main/datasets/set1/iris_kaggle.csv')

iris_setosa=iris.iloc[1:51]
iris_versi=iris.iloc[52:101]
iris_virgin=iris.iloc[102:151]

iris_setosa.reset_index(inplace=True)
iris_versi.reset_index(inplace=True)
iris_virgin.reset_index(inplace=True)

iris_setosa=iris_setosa[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
iris_versi=iris_versi[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
iris_virgin=iris_virgin[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

iris_setosa_corr_matrix=iris_setosa.corr()
print("The correlation matrix for the species Iris Setosa is,")
print(iris_setosa_corr_matrix)
print('\n')

iris_versi_corr_matrix=iris_versi.corr()
print("The correlation matrix for the species Iris Versicolor is,")
print(iris_versi_corr_matrix)
print('\n')

iris_virgin_corr_matrix=iris_virgin.corr()
print("The correlation matrix for the species Iris Virginica is,")
print(iris_virgin_corr_matrix)
print('\n')