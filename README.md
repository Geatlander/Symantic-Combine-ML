# Symantic-Combine-ML
A short experiment in machine learning to compare how well different models could combine symantic features of inputs.

The hypothesis of this experiment was that, given two textual inputs which contained different features,
a recurrent neural network would be able to combine symantic and syntactic features to create a 'hybrid' text.

Three techniques were tested and compared- two variations of an LSTM network and a simple Markov model.


An example of this would be feeding in a text file of French text and one of English, with the network outputting
a combination of the two languages, maybe something like:
'Il cannote pas fair hese travael.'

In this experiment, one text file that contained 'Le Comte de Monte Cristo' by Dumas was used, and another containing various English works. Let it be noted that the two files were decently sized and were relatively uniform in size.
In an attempt to produce that type of text, three different models were tested with various results. The first one was an LSTM Recurrent Neural Network, renowned for being able to extract features from large samples of text (http://karpathy.github.io/2015/05/21/rnn-effectiveness/). Skeleton code from this post: https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/ was used. Also, here is this guy's github repo: https://github.com/ChunML/text-generator.

The LSTM design referenced is proven to work effectively in text generation. When trained on some of Tolkien's works, the below excerpt was outputted:

Epoch 106
Epoch 1/1
41054/41054 [==============================] - 636s - loss: 0.4283     
� (and it would be safe.'

`What are the Riders? But'' asked Pippin. 'Here we must endure the Withywindle came into
the wreth of it are irpe; but if either the ores of the Valar, the Elder of Lórien, that if you did to do? ' she said. `We have been gently dart.' But he did not tell
me anxious now, perhaps. Within will happen in the woods of Shire-Man.'

Then some may walk free, and before them was scored and felt that it was the fountain of the stream above them. As soon as the council of th

---------------------------------------------------------------------------------------------------------------------------------

In order to combine the features of inputs in a similar way, then, the idea of training the network on one file, and then another, followed by an output query, was forumlated. To do this, a new class was created to encapsulate each piece of data, along with some other editions to the code. In the end, this approach didn't work so well. The network seemed to output text mimicking whatever input it was trained last off of, never mixing up the two styles of text. It was capable, strangely, of outputtting French on one occasion and English on another.

Naturally, it seemed logical to chop up each file into 100 sub-arrays to be fed to the model.
These were fed to the model sequentially, like this:

1. Train net on 1/100 of French text
2. Train net on 1/100 of English text
3. Output query the net
4. Repeat from 1.


I thought that feeding the model smaller samples of each text input would allow the weights to adjust in unison. It also seemed logical to decrease the learning rate in order to allow more liberal guesses. There were glimmers of hope in this attempt- once the network outputted something like 'Ce the the the...', which is in fact a combination of English and French words, just not an amalgamation. Occasionally, English words would be outputted with e's appended to the end, which gives a glimpse of what we were looking for in the first place. After hours of training though, the network was able to differentiate between the two and again two clear languages were being outputted.

After it was clear that this wasn't going to work, all kinds of different hyperparameters were tried in order to increase liberal learning. Dropout layers with a rate of 1/2 was used to prevent overfitting to any particular style, as well as varying temperatures (getting as high as 10 in experimentation), and a low learning rate that I set to increasingly low amounts. In the end, the model was still able to differentiate between two distinct styles of text, rather impressively. I concluded conclusively that an LSTM approach with two distinct input files is incapable of creating a hybrid-style text.

But the search was not done.
The next attempt was a simple Markov model, the code obtained from my brother's github gist: https://gist.github.com/davidcox143/8573063ef4beb762b167ed513ea6c49b. It was made specifically for text generation given an input file, which suited this purpose perfectly. For this attempt, the input files of both English and French were concatenated and the lines were shuffled randomely, so that the format of the document now looked something like:

French line

English line

French line

French line

English line

Upon querying the markov model, this kind of text was generated:

 mots pouvait good l'anéantis.» mantle lovely to hard de lord were l'honneur cursed And [2] au de Steingerd. il He the first; must dit subordonné état laisse stood ce tout chambre de one Derga. His woodbine, liberté Room no comme blow place him trois entendre le that. été yield A avec Pilon. the his "I by lui-même; overthrown n'oubliez nous Thy triumph, s'être à by it et of all la reft si vous secret jeune Ingcel. les alla then la in first à that Ah, vide. véritable, de ciel the the thy that know que dans Thorveig's les son «Et Angleterre: les that. of que par simple vingt une contrebandier As behind; o'er ye l'accolade: end die it his seems stand que trois qui shall eût de advienne is recette Thereupon they waters its -- bas, not; three in pas?»
 
This, admittedly, is pretty good. There isn't any 'mixing' between the words, and of course symantics aren't captured at all as the model is incapable of learning. However, this is the closest that we have gotten and it looks great. It also outputted instantly as opposed to the LSTM, which took close to a full day of training.


But let's return to the LSTM for a moment, because maybe we can try this kind of concatenation approach with it. With the (mostly) unmodified LSTM code by ChunML, we can use this same random concatenation approach as the markov model and train it as a normal text file.

In the end, the results for this were rather disappointing but I guess were to be expected-

Steingerd, "But now what was God is over and do sit trust, O not the holm he scurned,\n", 'and the shame-dolk of men, that shield-wall keep,\n', '\n', 'accompagnaient le gouverneur.\xc2\xbb\n', '\n', '   Amber and honey sweet.\n', "cette fois, il n'y a pas d'inconvence le vent \xc3\xa0 moi l'aidi\xc3\xa9e; puisque nous ne pouvons repartir\n", "l'homme.\n", '\n', 'sera, tout en buvourit.\n', 'of life despairing. -- No light thing that,\n', 'My love, used, it is for and for mine and the Saxon smittle, Cormac made up and spriad: but The archbishop saw the light of the house. Liken thouth the house of the hoary rain,\n', "--Sur ma m\xc3\xa9decinting, Edmond, consignation, mon capitaine au l\xc3\xa9gatif enca chancelait devant comme pr\xc3\xa8s de la tranture, et vous comprenez de Marseille. Il pronon\xc3\xa9 qu'il s'est bont\xc3\xa9 de francher en si, comme cette fois, rapport\xc3\xa9s de la conversation du cachot,\n', 'to the wave. He dark, - his helm were made?\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n'

It was just like what the input was- a line of one language followed by a line of the other. Even managing to mix the words together within the lines would have barely rivaled the Markov model with a significantly slower output time.

As far as concluding that this is impossible, it may well be possible.
All that is known now is that after adjusting multiple hyperparameters to even extreme limits, there is little to no improvement and no promise that there will be with an LSTM network- that's just not how they work. Perhaps there is another model out there better suited for this, but in the meantime the Markov model remains the best choice for this end. 



--Note: the source files for the models are as follows:

Markov- markov.py

First LSTM attempt by splitting text- morphgen.py

Second LSTM attempt by concatenating text- concatgen.py





