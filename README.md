## **1 Decision Trees with Categorical Attributes**
This part uses the adult data set (https://archive.ics.uci.edu/ml/datasets/Adult) from the
UCI Machine Learning Repository to predict whether the income of an individual exceeds 50K per
year based on 14 attributes. For this part, the attribute fnlwgt should be dropped and the following
attributes should be taken into consideration:
<table style="text-align:center" >
        <tr>
            <td>Attribute</td>
            <td>Description</td>
        </tr>
        <tr>
            <td>age</td>
            <td>age group</td>
        </tr>
        <tr>
            <td>workclass</td>
            <td>type of employment</td>
        </tr>
        <tr>
            <td>education</td>
            <td>level of education reached</td>
        </tr>
        <tr>
            <td>education-num</td>
            <td>number of education years</td>
        </tr>
        <tr>
            <td>marital-status</td>
            <td>type of maritals status</td>
        </tr>
        <tr>
            <td>occupation</td>
            <td>occupation domain</td>
        </tr>
        <tr>
            <td>relationship</td>
            <td>type of relationship involved</td>
        </tr>
        <tr>
            <td>race</td>
            <td>social category</td>
        </tr>
        <tr>
            <td>sex</td>
            <td>male or female</td>
        </tr>
        <tr>
            <td>capital-gain</td>
            <td>class of capital gains</td>
        </tr>
        <tr>
            <td>capital-loss</td>
            <td>class of capital losses</td>
        </tr>
        <tr>
            <td>hours-per-week</td>
            <td>category of working hours</td>
        </tr>
        <tr>
            <td>native-country </td>
            <td>country of birth</td>
        </tr>
    </table>

1. Read the data set and compute: (a) the number of instances, (b) a list with the
attribute names, (c) the number of missing values, (d) a list of the attribute names with at
1 least one missing value, and (e) the percentage of instances corresponding to individuals whose
education level is Bachelors or Masters (real number rounded to the first decimal digit).
2. Drop all instances with missing values. Convert all attributes (except the class) to
numeric using one-hot encoding. Name the new columns using attribute values from the original
data set. Next, convert the class values to numeric with label encoding.
3. Build a decision tree and classify each instance to one of the <= 50K and > 50K
categories. Compute the training error rate of the resulting tree.


## **2 Cluster Analysis**
This part uses the wholesale customers data set (https://archive.ics.uci.edu/ml/datasets/
wholesale+customers) from the UCI Machine Learning Repository to identify similar groups of customers based on 8 attributes. For this part of the coursework, the attributes Channel and Region
should be dropped. Only the following 6 numeric attributes should be considered:
<table style="text-align:center" >
        <tr>
            <td>Attribute</td>
            <td>Description</td>
        </tr>
        <tr>
            <td>Fresh</td>
            <td>Annual expenses on fresh products.</td>
        </tr>
        <tr>
            <td>Milk </td>
            <td>Annual expenses on milk products.</td>
        </tr>
        <tr>
            <td>Grocery </td>
            <td>Annual expenses on grocery products.</td>
        </tr>
        <tr>
            <td>Frozen</td>
            <td>Annual expenses on frozen products.</td>
        </tr>
        <tr>
            <td>Detergent</td>
            <td>Annual expenses on detergent products.</td>
        </tr>
        <tr>
            <td>Delicatessen</td>
            <td>Annual expenses on delicatessen products.</td>
        </tr>
    </table>

1. Compute the mean, standard deviation, minimum, and maximum value for each
attribute. Round the mean and standard deviation to the closest integers.

2. Divide the data points into k clusters, for k ∈ {3, 5, 10}, using kmeans and agglomerative hierarchical clustering. Because the performance of kmeans (e.g. number of iterations)
is significantly affected by the initial cluster center selection, repeat 10 executions of kmeans for
each k value. Next, standardize each attribute value by subtracting with the mean and then
dividing with the standard deviation for that attribute. Repeat the previous kmeans and agglomerative hierarchical clustering executions with the standardized data set. Identify which run
resulted in the best set of clusters using the Silhouette score as your evaluation metric. Visualize
the best set of clusters computed in the previous question. For this, construct a scatterplot for
each pair of attributes using Pyplot. Therefore, 15 scatter plots should be constructed in total.
Different clusters should appear with different colors in each scatter plot. Note that these plots
could be used to manually assess how well the clusters separate the data points.


## **3 Mining Text Data**
This part uses the Coronavirus Tweets NLP data set from Kaggle https://www.kaggle.com/
datatattle/covid-19-nlp-text-classification to predict the sentiment of Tweets relevant to
Covid. The data set (Corona NLP test.csv file) contains 6 attributes:
<table style="text-align:center" >
        <tr>
            <td>Attribute</td>
            <td>Description</td>
        </tr>
        <tr>
            <td>UserName</td>
            <td>Anonymized attribute.</td>
        </tr>
        <tr>
            <td>ScreenName</td>
            <td>Anonymized attribute.</td>
        </tr>
        <tr>
            <td>Location </td>
            <td>Location of the person having made the tweet.</td>
        </tr>
        <tr>
            <td>TweetAt</td>
            <td>Date.</td>
        </tr>
        <tr>
            <td>OriginalTweet</td>
            <td>Textual content of the tweet.</td>
        </tr>
        <tr>
            <td>Sentiment</td>
            <td>Emotion of the tweet.</td>
        </tr>
    </table>

Because this is a quite big data set, existing vectorized (pandas) functions should be particularly useful
to effectively perform the various tasks with a typical personal computer. In this way, you will be able
to run your code in few seconds. Otherwise, running your code might require a significant amount of
time, e.g. in the case where for loops are used for accessing all elements of the data set. Further, you
are expected to use raw Python string functions for text processing operations.

1. Compute the possible sentiments that a tweet may have, the second most popular
sentiment in the tweets, and the date with the greatest number of extremely positive tweets.
Next, convert the messages to lower case, replace non-alphabetical characters with whitespaces
and ensure that the words of a message are separated by a single whitespace.
2. Tokenize the tweets (i.e. convert each into a list of words), count the total number
of all words (including repetitions), the number of all distinct words and the 10 most frequent
words in the corpus. Remove stop words, words with ≤ 2 characters, and reduce each word to
its stem. You are now able to recompute the 10 most frequent words in the modified corpus.
What do you observe?
4. This task can be done individually from the previous three. Store the coronavirus tweets.py corpus in a numpy array and produce a sparse representation of the term document matrix with a CountVectorizer. Next, produce a Multinomial Naive Bayes classifier
using the provided data set. What is the classifier’s training accuracy? A CountVectorizer allows
limiting the range of frequencies and number of words included in the term-document matrix.
Appropriately tune these parameters to achieve the highest classification accuracy you can.




