
<p align = "middle">
  <img src="../master/AI-Poster.png"/>
  </p>
 
# Identifying Different Writing Styles in a Document Intrinsically Using Stylometric Analysis ✍️ 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2538334.svg)](https://doi.org/10.5281/zenodo.2538334)


[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](#)
[![GitHub Forks](https://img.shields.io/github/forks/harismuneer/Writing-Styles-Classification-Using-Stylometric-Analysis.svg?style=social&label=Fork&maxAge=2592000)](https://www.github.com/harismuneer/Writing-Styles-Classification-Using-Stylometric-Analysis/fork)
[![GitHub Issues](https://img.shields.io/github/issues/harismuneer/Writing-Styles-Classification-Using-Stylometric-Analysis.svg?style=flat&label=Issues&maxAge=2592000)](https://www.github.com/harismuneer/Writing-Styles-Classification-Using-Stylometric-Analysis/issues)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat&label=Contributions&colorA=red&colorB=black	)](#)


## Abstract

In this project, we have developed **an intelligent system that takes a single document and classifies different writing styles within the document using stylometric analysis**. The classification is done using **K-Means Clustering (an unsupervised machine learning method)**. First, the document is divided into chunks of text using a standard chunk size (a chunk is comprised of some fixed number of sentences). Then for each chunk of text, a vector of **stylometric features** is computed. After that the chunks are clustered using their vectors of stylometric features. Thats where unsupervised machine learning comes into play. **The chunks with same style are clustered together**. Hence, the number of clusters made correspond to the number of different writing styles that the document has. In this approach, the value of K is determined using Elbow Method. We also ran an experiment for a document with two writing styles and our system was successfully able to identify that the document had two different writing styles. Our approach separates out text with same style from that of different style and can be used to detect plagiarism also.

## Background

The study of measurable features of literary style, such as sentence length, readability scores, vocabulary richness and various frequencies (of words, word lengths, word forms, etc.), has been around at least since the middle of the 19th century, and has found numerous practical applications of interesting problems in the modern era of Artificial Intelligence and Machine Learning. **Study of a literary style of a document is called Stylometry.**

Stylometry grew out of earlier techniques of analyzing texts for evidence of authenticity, author identity, and other questions. The development of computers and their capacities for analyzing large quantities of data enhanced this type of effort by orders of magnitude. In the current era of research, Stylometry is widely used in the problems of intrinsic plagiarism detection, genres separation, authorship verifications and authorship attribution, author gender detections and many more. However, the main task is to classify different writing styles from the text, which can further be used to solve above-mentioned problems.

## Introduction

**Our system determines variations in writings in a text document. These variations can be due to different authors or different genres of writing for example stories, research papers, dramas play etc.**

Other approaches on **Intrinsic Plagiarism Detection** (essentially different writing styles detection) needed a large corpus of texts of different authors to train their models to know which text belonged to which author. They learnt the writing styles of each author and then given a document with an author they predicted whether that author plagiarised work of some other author or not. While our approach doesn’t need training on a large corpus of texts. It just extracts the essence of text style of each chunk of text using stylometric features and then groups together the chunks that have same writing styles. This process is repeated for every new document.

In this report, first we elaborate our whole methodology including the features selection and data preprocessing. It also includes the machine learning method used. After that in order to demonstrate our approach an experiment is run on a document with two different writing styles. The results are explained and some limitations of our work are also presented. At the end we conclude our report.

## Methodology

First, the document is divided into chunks of text using a standard chunk size window. Then for each chunk of text a vector of stylometric features is computed. After that the chunks are clustered using their vectors of stylometric features. Thats where unsupervised machine learning comes into play. The chunks with same style are clustered together.

### 1) Data Set Selection:

We selected our data set from the internet. http://textfiles.com/stories/is an online repository, which encompasses prodigious set of stories ranging from different authors to different difficulty levels. While we want to cluster different literary styles, we have used this data set for now to perform clustering. This dataset’s main purpose is just demonstration of our approach. **Our system can be run on any document.**
 
### 2) Features Selection:

The heart of our system lies in feature extraction. We have to use such features which inherit the style of that text, hence we carefully crafted features for our project from the ones we studied during literature review. In order to distinguish a chunk of text on the basis of its literary style we first needed to define its writing style. **A literary style spans a lot of things but we rather focused on three major ones: Lexical Features, Vocabulary Richness Features and Readability Scores.** These include features like Shannon Entropy and Simpson's Index. Simpson’s index stems from the concept of biodiversity. We used that in our project as we wanted to measure the diversity of a text. We used python to code these features. Following is the list of features we have extracted:

### Lexical Features:

* 1.	Average Word Length
* 2.	Average Sentence Length By Word
* 3.	Average Sentence Length By Character
* 4.	Special Character Count
* 5.	Average Syllable per Word
* 6.	Functional Words Count
* 7.	Punctuation Count

These are the most basic features one can extract from the text. These features tell us about the structure of the text. For example averages of different counts like word lengths, special characters, punctuations and functional words etc. Functional words are used to express grammatical relationships among other words within a sentence. Secondly, if a word has more syllables then it is most likely to be a difficult word (although not necessary). Avg Syllable per word being the measure of complexity, is used in calculations of many other features related to readability scores described in the sections ahead. Punctuation Countand Special Character Count are straight forward ways to differentiate different genres. For example narrative story and research paper.

### Vocabulary Richness Features:

Many quantitative studies rely on the concept of vocabulary richness. A text has low vocabulary richness if the same limited vocabulary is repeated over and over again, while it has high vocabulary richness if new words continually appear. In essence, these features tell us about the diversity and richness of the vocabulary used in the text.

* 1.	Hapax Legomenon
* 2.	Hapax DisLegemena
* 3.	Honores R Measure
* 4.	Sichel’s Measure
* 5.	Brunets Measure W
* 6.	Yules Characteristic K
* 7.	Shannon Entropy
* 8.	Simpson’s Index

### Readability Scores:

Readability is the ease with which a reader can understand a written text. Readability is more than simply legibility—which is a measure of how easily a reader can distinguish individual letters or characters from each other. Features for readability stems from the field of linguistics and researchers have frequently used linguistics’ laws (e.g zipfs law) and lemmas to pull out the currently used features to calculate readability scores of text in the modern computer science. Following is the list of features we are using.

* 1.	Flesch Reading Ease
* 2.	Flesch-Kincaid Grade Level
* 3.	Gunning Fog Index
* 4.	Dale Chall Readability Formula
* 5.	Shannon Entropy
* 6.	Simpson's Index


### 3) Data Pre-processing:

After downloading the data setfromtextfiles.com which consists of different text files of different authors and from different genres, we took a children story and a research paper from it for proof of concept. A document is then divided into small chunks. Here determining the size of chunk was a challenge for us. If it was too large then we won't be able to extract the crux of different passages. Had it been too small it would have lost the significance. Hence we went for the average i.e 10 sentences (it can be changed according to needs too) Now first of all the lexical features are computed for each chunk. Then for all other features we removed punctuations and special characters and performed tokenization (because lexical features use punctuations and special characters).


### 4) Machine Learning Algorithm:

As we are using unsupervised learning approach to cluster our data, we have used the most famous algorithm in this domain i.e. K-Means algorithm for our purpose.

### 5) PCA and Data Visualization:

As mentioned earlier, we have calculated almost 20 features. After that K-Means algorithm is run on the vectors of all chunks and centroids are identified. Now at this stage the number of centroids correspond to the number of different writing styles identified and this was what our system was meant to do but in order to visually see those clusters we had to convert our 20 dimensional vector into a 2D vector using Principal Component Analysis which extracted the essence from that 20D vector and converted it into a 2D vector. We then plot these vectors and color those chunks same which were grouped together under a centroid by K-Means. This way the chunks with different styles are visualized further strengthening our results.
 
## Experimental Settings

For the proof of concept, we chose two documents. One is a story named Jim(Story) while the other document is a research paper named AuthAttr(Paper). Now we merge these two documents into one and use that document for experiments. Since, this new document contains two different writing styles (one of a story and one of a research paper) hence, our system should essentially identify that this new document has two writing styles.

### K - Means

We used K-Means algorithm to identify Kdifferent centroids in a text having different writing styles. Each centroid spans those chunks which have the same writing style. Hence the number of centroids correspond to the different number of writing styles that a document has.

### Value of K:

We can choose the number of clusters by visually inspecting user data points first using their vector of stylometric features, but we soon realized that there is a lot of ambiguity in this process for all except the simplest data sets. This is not always bad, because we are doing unsupervised learning and there's some inherent subjectivity in the labeling process. Still tt is necessary to know the value of K before hand to run K-means effectively.

We used the Elbow method to find the optimal value of K.

### Elbow method:

The elbow method is described below:

First of all, compute the sum of squared error (SSE) for some values of k (for example 2, 4, 6, 8, etc.). The SSE is defined as the sum of the squared distance between each member of the cluster and its centroid. 

If we plot k against the SSE, we will see that the error decreases as k gets larger; this is because when the number of clusters increases, distortion gets smaller. The idea of the elbow method is to choose the k at which the SSE decreases abruptly. This produces an "elbow effect" in the graph, as can be seen in the following picture:

![elbow](../master/Images/elbow.jpg)

In this case, the most suitable value for K is k = 2.

Take into account that the Elbow method is an heuristic and, as such, it may or may not work well in usr particular case. Sometimes, there are more than one elbow, or no elbow at all. In those situations us usually end up calculating the best k by evaluating how well k-means performs in the context of the particular clustering problem us are trying to solve.

### Parameter Tuning of K-Means:

We have used K-Means algorithm from sklearn library of python. At first we selected the value of K from Elbow method but there are other parameters whose values are very important to be well taken care of. After running multiple experiments we found out following parameter values to be best in our scenario:

#### n_init:

As K means is heuristic based it depends upon the start seeds value of centroids we place at the start of the algorithm. It may stuck at local optima so We used n_init value=	10. It's basically randomly re initializes the centroids. So K-Means will be run n_init number of times with different centroid seeds. The final results will be the best output of n_initconsecutive runs in terms of inertia.

#### Max_iter:

It is the maximum number of iterations K-Means algorithm for a single run. We used max_iter = 500 With minimum tolerance for convergence.

#### n_Jobs:

The number of jobs to use for the computation. This works by computing each of the n_init runs in parallel. We have used n_jobs = -1 Which will utilize all the cpu’s available on the host machine.

## Results:

We ran this experiment with a document file containing two different styles (one of a story and one of a research paper). And our system was correctly able to detect those two different writing styles and also color those chunks with the same writing style. The elbow 
method successfully returned K = 2 which was our goal.


A document containing texts with 2 different writing styles (a story and a research paper) are clearly distinguished indicating the correctness of our approach:

<p align="middle">
  <img src="../master/Images/results.jpg"/>
 </p>
 
## Limitations

As PCA smudges the higher dimensional vector into a 2D vector, so there is a high possibility of some loss of significance during the conversion process. Due to this there is a possibility that after plotting the clusters don’t seem distinguishable as expected due to the loss of information while doing PCA. This is not as such a limitation to our system, since our system correctly identifies the number of writing styles through the value of K which is enough to proof the correctness of our approach but if and only if there is a need to visually see those clusters then it is only possible by converting the high dimensional vectors using PCA to a 2D one.

In a scenario where the document is written by a single author, one may assume that in this case there should be only one cluster implying one writing style. However, this is not the case every time. As we know each paragraph written even by the same author has a bit different writing style than its other counterparts like e.g there are some minor differences in lexical features (number of punctuation marks and others). Also there might be a poetry in paragraphs, mathematical equations etc. So our method will create clusters according to the writing style within that document. But the major difference in such a setting is the distance between those clusters. The more different the writing styles are the more far away the clusters are from each other. In the case of a document with the same writing style, even if some clusters are made, the distance between those clusters should be small as compared to the case if the document had really different writing styles. Hence this is the crucial essence which makes our algorithm excel because it not only identifies one part of the story i.e number of writing styles but also shows how different those writings styles are from each otherwhich is implied by the distance between the clusters.

## Conclusion

To sum up:

* 1.	Our system takes a document.
* 2.	Divides it into chunks of 10 sentences.
* 3.	Computes stylometric features for each chunk.
* 4.	Then uses the elbow method on these vectors to identify the value of centroids K.
* 5.	The value of K corresponds to the number of different writing styles the document had.
* 6.	In order to visualize the clusters, PCA is used to convert the high dimensional features vector to a 2D one and then the chunks are plotted.
* 7.	The chunks with same style are grouped under one centroid with same color, hence implying the number of writing styles implied in that document.

The heart of our approach is correctly extracting the style from the chunk which is successfully achieved using a mix of different categories of linguistic features like lexical, vocabulary richness and readability scores. Our method is repeated every time for a new document. Since tt identifies the different writing styles in that document, hence our approach can also be used to detect plagiarism.

-------------------------------------
## Code
The complete code for experimentation is also provided. We have open-sourced it to facilitate research in such an exciting domain.

-------------------------------------
## Citation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2538334.svg)](https://doi.org/10.5281/zenodo.2538334)

If you want to cite our work, kindly click the above badge for more information regarding the complete citation for this research work and diffferent citation formats like IEEE, APA etc.

-------------------------------------

## Authors

You can get in touch with us on our LinkedIn Profiles:

#### Hassaan Elahi

[![LinkedIn Link](https://img.shields.io/badge/Connect-Hassaan--Elahi-blue.svg?logo=linkedin&longCache=true&style=social&label=Connect)](https://www.linkedin.com/in/hassaan-elahi/)

You can also follow my GitHub Profile to stay updated about my latest projects: [![GitHub Follow](https://img.shields.io/badge/Connect-Hassaan--Elahi-blue.svg?logo=Github&longCache=true&style=social&label=Follow)](https://github.com/Hassaan-Elahi)

#### Haris Muneer

[![LinkedIn Link](https://img.shields.io/badge/Connect-harismuneer-blue.svg?logo=linkedin&longCache=true&style=social&label=Connect)](https://www.linkedin.com/in/harismuneer)

You can also follow my GitHub Profile to stay updated about my latest projects: [![GitHub Follow](https://img.shields.io/badge/Connect-harismuneer-blue.svg?logo=Github&longCache=true&style=social&label=Follow)](https://github.com/harismuneer)

If you liked the repo then please support it by giving it a star ⭐!

## Contributions Welcome
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](#)

If you find any bug in the code or have any improvements in mind then feel free to generate a pull request.

## Issues
[![GitHub Issues](https://img.shields.io/github/issues/harismuneer/Writing-Styles-Classification-Using-Stylometric-Analysis.svg?style=flat&label=Issues&maxAge=2592000)](https://www.github.com/harismuneer/Writing-Styles-Classification-Using-Stylometric-Analysis/issues)

If you face any issue, you can create a new issue in the Issues Tab and we will be glad to help you out.

## License
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a> [![MIT](https://img.shields.io/cocoapods/l/AFNetworking.svg?style=style&label=License&maxAge=2592000)](../master/LICENSE)

The content of this project is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>, and the underlying source code used is licensed under the MIT license. 

Copyright (c) 2018-present, Hassaan-Elahi, harismuneer                                                        
