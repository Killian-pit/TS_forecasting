from keras.models import Sequential
from keras.layers import Embedding,Reshape,Dense,Concatenate,Flatten,Dropout,SpatialDropout1D
# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras import optimizers

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler,PowerTransformer
from sklearn.linear_model import Ridge
from sklearn.base import TransformerMixin 
import numpy as np 


class MultiColumnLabelEncoder:
	'''
	description :
	create a sklearn like multi column encoder for categorical features 
	'''
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


def create_mlp(layers_list=[1024,1024,512,128]):
	'''
	description : generate a basic regression mlp with
	both embedding entries for categorical features and 
	standard inputs for numerical feaatures
	params:
	layers_list : list of layers dimensions 
	output :
	compiled keras model  
	'''

    # define our MLP network
    layers = []
    output_num = []
    inputs = []
    output_cat = []
    
    #numerical data part 
    if len(numeric_features)>1:
        for num_var in numeric_features:
            input_num = Input(shape=(1,), name='input_{0}'.format(num_var))
            inputs.append(input_num) 
        output_num = Concatenate()(inputs)
    else:
        input_num = Input(shape=(1,), name='input_{0}'.format(numeric_features[0]))
        inputs.append(input_num)    
        output_num= input_num
    
    # create an embedding for every categorical feature 
    for categorical_var in categorical_features :
        no_of_unique_cat  = df[categorical_var].nunique()
        embedding_size = min(np.ceil((no_of_unique_cat)/2), 100 )
        embedding_size = int(embedding_size)
        vocab  = no_of_unique_cat+1
        #### functionnal loop 
        input_cat = Input(shape=(1,), name='input_{0}'.format(categorical_var))
        embedding =  Embedding(vocab, embedding_size,name='embedding_{0}'.format(categorical_var))(input_cat)
        embedding = SpatialDropout1D(0.2)(embedding)
        vec = Flatten(name='flatten_{0}'.format(categorical_var))(embedding)
        output_cat.append(vec)
        inputs.append(input_cat)     
    output_cat = Concatenate()(output_cat)
        
    dense = Concatenate()([output_num, output_cat])
    
    for i in range(len(layers_list)):
        dense = Dense(layers_list[i], name='Dense_{0}'.format(str(i)),activation='relu')(dense)
        dense = Dropout(.1)(dense)

    dense2 = Dense(1, name='Output', activation='relu')(dense)
    model = Model(inputs,dense2)
    
    opt = optimizers.Nadam(lr=1e-3)
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
    return model

def transform_to_supervised(df, previous_steps=1, forecast_steps=1, dropnan=True):
    # Transform time series data to supervised Tshifted time series 
    col_names = df.columns
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(previous_steps, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (col_name, i)) for col_name in col_names]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, forecast_steps):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s(t-0)' % col_name) for col_name in col_names]
        else:
            names += [('%s(t+%d)' % (col_name, i)) for col_name in col_names]

    # put all the columns together into a single aggregated DataFrame
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg
    
def create_cat_reg(layers_list=[256,256,256,128]):
    '''
    description :
    categorical only  only regressor 
    '''
    layers = []
    output_num = []
    inputs = []
    output_cat = []
    
    for categorical_var in categorical_features :
        no_of_unique_cat  = df[categorical_var].nunique()
        embedding_size = min(np.ceil((no_of_unique_cat)/2), 100 )
        embedding_size = int(embedding_size)
        vocab  = no_of_unique_cat+1
        #### functionnal loop 
        input_cat = Input(shape=(1,), name='input_{0}'.format(categorical_var))
        embedding =  Embedding(vocab, embedding_size,name='embedding_{0}'.format(categorical_var))(input_cat)
        embedding = SpatialDropout1D(0.1)(embedding)
        vec = Flatten(name='flatten_{0}'.format(categorical_var))(embedding)
        output_cat.append(vec)
        inputs.append(input_cat)     
        
    dense = Concatenate()(output_cat)
        
    for i in range(len(layers_list)):
        dense = Dense(layers_list[i], name='Dense_{0}'.format(str(i)),activation='linear')(dense)
        dense = Dropout(.1)(dense)

    dense2 = Dense(1, name='Output', activation='relu')(dense)
    model = Model(inputs,dense2)

    
    opt = optimizers.Nadam(lr=1e-3)
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
    return model
