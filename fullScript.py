# Database Section
import MySQLdb
import numpy as np

db = MySQLdb.connect(host="localhost",user="foo",passwd="bar",db="stuffs");
cursor = db.cursor();
cursor.execute("SELECT COUNT(*) FROM tweets;");
tweetCount = int(cursor.fetchone()[0]);
print "Tweet Count : %i " % tweetCount;

cursor.execute("SELECT id, text, IF (LENGTH(in_reply_to_status_id_str) = 0, false,true), IF (LENGTH(in_reply_to_user_id_str) = 0, false,true),LENGTH(user_name),LENGTH(user_screen_name), user_description, user_protected, user_followers_count, user_friends_count, user_listed_count, user_favourites_count, user_verified, user_statuses_count, entities_hashtags, entities_symbols, entities_user_mentions, entities_urls, retweet_count FROM tweets;");
results = cursor.fetchall();

# Get all the popular words from the tweets
#words = {};
#index = 0;
#for row in results:
#    wordsFromTweet = row[1].split(' ');
#    index = index + 1;
#	for word in wordsFromTweet:
#		if word not in words:
#            words[word] = 1;
#        else:
#            words[word] += 1;

#print(str(len(words)) + " words found");

# Sort the words by popularity
#import operator
#sortedWords = sorted(words.items(), key=operator.itemgetter(1),reverse=True)

# Check to make sure the word is used over the threshold
# before considering it a popular word

#thresholdWords = 200;
#frequentWords = [];
#for i in range(len(sortedWords)):
#    if (sortedWords[i][1] > thresholdWords):
#        frequentWords.append(sortedWords[i][0]);
#    else:
#        break;
#print("Popular Words: " + str(len(frequentWords)));

arrayOfTweets = [];

featureCount = 20; #(20 + len(frequentWords)); # don't use frequent words

for row in results:
    dataPoint = [None] * featureCount;
    wordsInTweet = row[1].split(' ');
    dataPoint[0] = float(len(wordsInTweet)); # word count of tweet text
    dataPoint[1] = float(len(row[1])); # character count of tweet text
    dataPoint[2] = float(row[2]); # is in reply to status
    dataPoint[3] = float(row[3]); # is in reply to user
    dataPoint[4] = float(row[4]); # character count of user name
    dataPoint[5] = float(row[5]); # character count of screen name
    dataPoint[6] = float(len(row[6].split(' '))); # word count of description
    dataPoint[7] = float(len(row[6])); # character count of description
    dataPoint[8] = float(row[7]); # is the user protected
    dataPoint[9] = float(row[8]); # user follower count
    dataPoint[10] = float(row[9]); # user friends count
    dataPoint[11] = float(row[10]); # user listed count (how many times they are referenced)
    dataPoint[12] = float(row[11]); # user favourites count (how many tweets they have favourited)
    dataPoint[13] = float(row[12]); # is user verified
    dataPoint[14] = float(row[13]); # user statuses count
    dataPoint[15] = float(row[14]); # hashtag count in tweet
    dataPoint[16] = float(row[15]); # symbol count in tweet
    dataPoint[17] = float(row[16]); # user mentions count in tweet
    dataPoint[18] = float(row[17]); # url count in tweet
    
    #for index in range(len(frequentWords)):
    #    dataPoint[19+index] = 1.0 if (frequentWords[index] in wordsInTweet) else 0.0;
    
    # class label definition
    if (row[18] == 0):
        dataPoint[featureCount-1] = 0; # class 0
    elif (row[18] >= 1 and row[18] <= 50):
        dataPoint[featureCount-1] = 1; # class 1
    elif (row[18] > 50 and row[18] <= 1000):
        dataPoint[featureCount-1] = 2; # class 2
    else:
        dataPoint[featureCount-1] = 3; # class 3
    arrayOfTweets.append(dataPoint);
print(len(arrayOfTweets));

# function to normalize the data in the test/train data
def normalizeData(data):
    for feature in range(len(data[0])):
        if (feature == 19):
            continue;
        minX = data[0][feature];
        maxX = data[0][feature];
        for index in range(len(data)):
            if (data[index][feature] < minX):
                minX = data[index][feature];
            elif (data[index][feature] > maxX):
                maxX = data[index][feature];

        # compute the denominator just once
        denominator = float(maxX - minX);
        #print(denominator);
        if (denominator == 0):
            continue;

        # normalize the data points
        for index in range(len(data)):
            data[index][feature] = float((float(data[index][feature]) - float(minX))) / float(denominator);

trainData = np.array(arrayOfTweets[0:70000]);
testData = np.array(arrayOfTweets[70001:]);

trainDataLabels = trainData[:,len(trainData[0])-1]
trainData = np.delete(trainData, len(trainData[0])-1, 1)

testDataLabels = testData[:,len(testData[0])-1]
testData = np.delete(testData, len(testData[0])-1, 1)

testDataNormalize = np.array(testData);
trainDataNormalize = np.array(trainData);
normalizeData(testDataNormalize);
normalizeData(trainDataNormalize);

# method to print out headers
def printHeader(headerString):
	dashString = "-" * len(headerString);
	borderString = "+-" + dashString + "-+";
	titleString = "| " + headerString + " |";
	print("");
	print(borderString);
	print(titleString);
	print(borderString);
	print("");

########################################
# Non - Normalized Data
########################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
printHeader("KKN-C Non Normalized Hyper-Parameter Search");

#Let's do some hyper-parameter searching
trainDataSplit1 = np.array(trainData[0:int(len(trainData)/2.0)]);
trainDataSplit2 = np.array(trainData[int(len(trainData)/2.0):]);
trainDataSplit1Labels = np.array(trainDataLabels[0:int(len(trainData)/2.0)]);
trainDataSplit2Labels = np.array(trainDataLabels[int(len(trainData)/2.0):]);
maxK = 1;
maxAccuracy = 0;
for k in range(1,39,2):
	knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
	knn.fit(trainDataSplit1, trainDataSplit1Labels)
	predictions = knn.predict(trainDataSplit2)
	accuracy = metrics.accuracy_score(trainDataSplit2Labels, predictions)
	print("k = " + str(k) + ", accuracy = " + str(accuracy));
	if (accuracy > maxAccuracy):
		maxK = k;
		maxAccuracy = accuracy;

print("Best k = " + str(maxK) + " with accuracy = " + str(maxAccuracy));

########
# Run KNN using the k Found
########

printHeader("KKN-C run on non-normalized data");
knn = KNeighborsClassifier(n_neighbors=maxK, metric="euclidean")
knn.fit(trainData, trainDataLabels)
predictions = knn.predict(testData)
accuracy = metrics.accuracy_score(testDataLabels, predictions)
cm = metrics.confusion_matrix(testDataLabels, predictions) # run the confusion matrix
print(cm)
print("Accuracy: " + str(accuracy));


#######
# PCA Non Normalized Data
#######
printHeader("Run PCA On Non-Normalized Data 15 Components");

from sklearn.decomposition import PCA
print(np.array([trainDataLabels]))
#trainDataNew = np.concatenate((trainData, np.array([trainDataLabels]).T),axis=1);

pca = PCA(n_components=15);
pca.fit(trainData);
transformedTrainData = pca.transform(trainData);
transformedTestData = pca.transform(testData);

print(pca.get_params());


printHeader("KKN-C PCA Data with Non-Normalized Train Data Hyper-Parameter Search");

#Let's do some hyper-parameter searching
trainDataSplit1 = np.array(transformedTrainData[0:int(len(transformedTrainData)/2.0)]);
trainDataSplit2 = np.array(transformedTrainData[int(len(transformedTrainData)/2.0):]);
trainDataSplit1Labels = np.array(trainDataLabels[0:int(len(transformedTrainData)/2.0)]);
trainDataSplit2Labels = np.array(trainDataLabels[int(len(transformedTrainData)/2.0):]);
maxK = 1;
maxAccuracy = 0;
for k in range(1,39,2):
	knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
	knn.fit(trainDataSplit1, trainDataSplit1Labels)
	predictions = knn.predict(trainDataSplit2)
	accuracy = metrics.accuracy_score(trainDataSplit2Labels, predictions)
	print("k = " + str(k) + ", accuracy = " + str(accuracy));
	if (accuracy > maxAccuracy):
		maxK = k;
		maxAccuracy = accuracy;

print("Best k = " + str(maxK) + " with accuracy = " + str(maxAccuracy));

printHeader("KKN-C PCA Data with Non-Normalized Train Data");
knn = KNeighborsClassifier(n_neighbors=maxK, metric="euclidean")
knn.fit(transformedTrainData, trainDataLabels)
predictions = knn.predict(transformedTestData)
accuracy = metrics.accuracy_score(testDataLabels, predictions)
cm = metrics.confusion_matrix(testDataLabels, predictions) # run the confusion matrix
print(cm)
print("Accuracy: " + str(accuracy));



########################################
# Normalized Data
########################################

printHeader("KKN-C Normalized Hyper-Parameter Search");

#Let's do some hyper-parameter searching
trainDataSplit1 = np.array(trainDataNormalize[0:int(len(trainDataNormalize)/2.0)]);
trainDataSplit2 = np.array(trainDataNormalize[int(len(trainDataNormalize)/2.0):]);
maxK = 1;
maxAccuracy = 0;
for k in range(1,39,2):
	knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
	knn.fit(trainDataSplit1, trainDataSplit1Labels)
	predictions = knn.predict(trainDataSplit2)
	accuracy = metrics.accuracy_score(trainDataSplit2Labels, predictions)
	print("k = " + str(k) + ", accuracy = " + str(accuracy));
	if (accuracy > maxAccuracy):
		maxK = k;
		maxAccuracy = accuracy;

print("Best k = " + str(maxK) + " with accuracy = " + str(maxAccuracy));

########
# Run KNN using the k Found
########

printHeader("KKN-C run on normalized data");
knn = KNeighborsClassifier(n_neighbors=maxK, metric="euclidean")
knn.fit(trainDataNormalize, trainDataLabels)
predictions = knn.predict(testDataNormalize)
accuracy = metrics.accuracy_score(testDataLabels, predictions)
cm = metrics.confusion_matrix(testDataLabels, predictions) # run the confusion matrix
print(cm)
print("Accuracy: " + str(accuracy));


#######
# PCA Normalized Data
#######
printHeader("Run PCA On Normalized Data 15 Components");

from sklearn.decomposition import PCA
print(np.array([trainDataLabels]))
#trainDataNew = np.concatenate((trainData, np.array([trainDataLabels]).T),axis=1);

pca = PCA(n_components=15);
pca.fit(trainDataNormalize);
transformedTrainData = pca.transform(trainDataNormalize);
transformedTestData = pca.transform(testDataNormalize);

print(pca.get_params());


printHeader("KKN-C PCA Data with Normalized Train Data Hyper-Parameter Search");

#Let's do some hyper-parameter searching
trainDataSplit1 = np.array(transformedTrainData[0:int(len(transformedTrainData)/2.0)]);
trainDataSplit2 = np.array(transformedTrainData[int(len(transformedTrainData)/2.0):]);
trainDataSplit1Labels = np.array(trainDataLabels[0:int(len(transformedTrainData)/2.0)]);
trainDataSplit2Labels = np.array(trainDataLabels[int(len(transformedTrainData)/2.0):]);
maxK = 1;
maxAccuracy = 0;
for k in range(1,39,2):
	knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
	knn.fit(trainDataSplit1, trainDataSplit1Labels)
	predictions = knn.predict(trainDataSplit2)
	accuracy = metrics.accuracy_score(trainDataSplit2Labels, predictions)
	print("k = " + str(k) + ", accuracy = " + str(accuracy));
	if (accuracy > maxAccuracy):
		maxK = k;
		maxAccuracy = accuracy;

print("Best k = " + str(maxK) + " with accuracy = " + str(maxAccuracy));

printHeader("KKN-C PCA Data with Normalized Train Data");
knn = KNeighborsClassifier(n_neighbors=maxK, metric="euclidean")
knn.fit(transformedTrainData, trainDataLabels)
predictions = knn.predict(transformedTestData)
accuracy = metrics.accuracy_score(testDataLabels, predictions)
cm = metrics.confusion_matrix(testDataLabels, predictions) # run the confusion matrix
print(cm)
print("Accuracy: " + str(accuracy));

