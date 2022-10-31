import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
import time


'''Project's purpose is to tokenize the text of all pitchfork reviews and train a deep neural network to match the tokenized
text with a rating'''


pd.options.display.max_rows = None


#All pitchfork reviews were scrapped from different genre webpages and saved to seperate xlsx documents.  The code below
#opens these diles and merges them into a single pandas data frame
df = pd.read_excel('C:\PF\PF.xlsx')
df2 = pd.read_excel('C:\PF\PFrock.xlsx')
df3 = pd.read_excel('C:\PF\PFfolk.xlsx')


df = df.append(df2, ignore_index=True)

#rows that contain n/a values are dropped
df = df.dropna()
#scores are converted from a string to numberic value
df['score'] = pd.to_numeric(df['score'])

#dataframe rows are shuffled to avoid bias in the testing and train split
df_shuffled = shuffle(df)

#data frame is split into two parts, one to train the data and another to test the trained model's accuracy
training_data = df_shuffled.iloc[:11200, :]
testing_data = df_shuffled.iloc[11200:, :]

#dataframe rows are converted into numpy arrays
training_sentences = np.array(training_data['text'])
training_labels = np.array(training_data['score'])

testing_sentences = np.array(testing_data['text'])
testing_labels = np.array(testing_data['score'])

#Tokenizer Params - top 70000 words will be tokenized, the max length of each review to be tokenized is 500 words, 0s will
#be applies at the end of the vector and words found outside of the tokenizer will be replaced with a <00V> token
vocab_size = 70000
max_length = 500
trunc_type = 'post'
oov_tok = '<OOV>'

#All words in all reviews are tokenized, these tokens are applied to all reviews to get vector values, these vectors are
#then padded with 0s
tokenizer = Tokenizer(num_words = vocab_size, oov_token= oov_tok)
tokenizer.fit_on_texts(training_sentences)

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, truncating=trunc_type)

#Convolution parameters - tokens are influenced by 20 bearby tokens to create context
batch_size = 1
timesteps = 20

#Model Params - Embedding: each word is given a vector within a 32 dimensional space which gives it greater than linear
#context in relation to all other words, 64 different convolutional filters are applied to each review matrix,
embedding_dim = 32
filters = 64
kernel_size = 7

#model parameters are set.  Tokenized reviews are embedded in multiple dimensions, convolutions are applied and a single
#linear dense (we are doing a linear regression model) is used to get a score output.
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
    #tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='linear')
])

print(model.summary())

#model is compiled using the mean absolute error (squared wasn't chosen to make the model less conservative), optimization
#function is adamax
model.compile(loss='mae', optimizer='adamax')

#Run Params - model is run in batches of 100 for 50 passes
epochs = 50
batch_size = 1000

#This runs the model and reports our accuracy at each successive epoch
history = model.fit(training_padded, training_labels, batch_size=batch_size, epochs=epochs, validation_data=(testing_padded, testing_labels))

#below code is testing the model against any text extered in the text variable (must be outside the dataset to be meaningful)
text = ['Will you hold me? Im scared. Its not that Im easily frightened. I can sit through the most gruesome horror movie, happily munching away on imitation butter-flavored popcorn while small intestines are strewn about the screen. I can watch news stories about mad cow disease while devouring sheeps brain soufflé. But listening to the latest Of Montreal record, all I can do is curl up in a ball, smile the sick, twisted smile of the damned, and nod my head up and down in a rhythmic fashion.If youre familiar with the work of Athens popsters Of Montreal, the disconcerting mix of a feeling of imminent implosion and a nagging urge to draw pictures of smiling bunny rabbits Im experiencing at the moment probably doesnt surprise you. Of Montreals music, and the bizarre artwork that accompanies it, plays like a surreal carnival-- it can be beautiful, it can be fun, and it can also be weird and creepy. Coquelicot, like most Of Montreal albums, is at times sublime and lovely, at times infuriatingly catchy, at times simply infuriating, at times overly twee, and at times seriously fucking scary. What sets this record apart from its predecessors, though, is a level of intricacy and detail that Of Montreal have never previously attained as a band. The songs on Coquelicot, though crammed full of saccharine pop hooks, display a level of complex structuring and arrangement that could put most pop records to shame.Of Montreals trademark hyperactivity, and melodic yet off-kilter sensibility is possibly at its peak on Coquelicot. Seemingly drawing as much from the English music hall tradition as from American pop acts like the Beach Boys, theres nothing else out there quite like the frenetic, utterly wacked-out pop these guys come up with.When the records at its best, the group incorporates more diverse elements into their music than ever before. Good Morning Mr. Edminton, Coquelicots opener, is a typical Of Montreal song in prime form. Fuzzed-out guitar, bouncy piano, and multitracked harmonies by frontman Kevin Barnes set the stage for a demented tale of kidnapping and working class struggle as told, of course, from the kidnappers point of view. The Peacock Parasols, which features a truly unforgettable, cryptic, and quite possibly misspelled lyric referring to P.P. icycles, goes from a pop song in warp drive to a dense, orchestral middle, and back.Though fast-paced pop is clearly the bread and butter of Coquelicot, its far from the only style to be found on this record. Theyre not playing around about the variety of whimsical verse thing. And sadly, this means the inclusion of the intolerable skit The Events Leading Up to the Collapse of Detective Dulllight, in which Kevin Barnes seeks to shatter your preconceived notions of reality by introducing a character whose name contains three consecutive ls. After a series of good songs, nothings quite as frustrating as hitting a 2-bit comedy routine in which some guy named Slocks writes a poem called The Cause of Gauze. And then reads it aloud.Besides annoying passages like these, the albums most challenging element is its length. At a solid 70 minutes, its almost impossible to endure an entire sitting. Had the filler been cut, this would easily be their best album yet, but repeated exposure to this stuff for that length of time cant be good for anybody. Of course, I dont have to worry, because Im a five-footed, owl-headed elf named Figgienewton! Uh-oh...']

predict_sequence = tokenizer.texts_to_sequences(text)
padded_predict = pad_sequences(predict_sequence, maxlen=max_length, truncating=trunc_type)

print(padded_predict)

prediction = model.predict(padded_predict)

print(prediction)
print(prediction.shape)







