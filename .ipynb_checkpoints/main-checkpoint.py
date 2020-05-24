# preprocessing pipeline
from sklearn.compose import ColumnTransformer

# here goes the program to show the use of utility functions useed previously 
s
categorical_transformer = Pipeline(steps=[
    #('imputer', SimpleImputer(strategy='most_frequent')),
    ('Multilabel',  MultiColumnLabelEncoder(categorical_features)),
])
numeric_transformer = Pipeline(steps=[
    #('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler()),
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

