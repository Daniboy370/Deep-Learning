# Deep-Learning

097200 - [Deep Learning, Theory and Practice](https://www.graduate.technion.ac.il/Subjects.Heb/?Sub=97200).

Syllabus : The course covers a variety of computational models and algorithms with an emphasis on deep learning
app computerized vision and language processing alongside convolution and their generalization capabilities.

- **Transfer Learning via pre-trained Inception V3** :

- Downloading from Kaggle the 'fruits-360' dataset, I reduced it into a smaller collection of 7 out of 360 fruits. I then froze the CNN base ("chop the head") and attached a trainable fully connected network such that the smaller dataset will be learnt 
![alt text](https://github.com/Daniboy370/Deep-Learning/blob/master/Homeworks/RNN_LSTM/output/inception_model.png)

Assigning smaller on-line flow of batch sizes to reduce RAM usage, and got the following results :
![alt text](https://github.com/Daniboy370/Deep-Learning/blob/master/Homeworks/RNN_LSTM/output/graph_table.png)

![alt text](https://github.com/Daniboy370/Deep-Learning/blob/master/Homeworks/RNN_LSTM/output/RNN_results.png)
