import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')
import pickle


def readData(name):
	data = pd.read_csv(name)
	return data

df = readData('2015-17_teamBoxScore.csv')
df2 = readData('2017-18_teamBoxScore.csv')

def informationAboutData(df):
	print('(Nombre de ligne, nombre de colonne)')
	print(df.shape)

informationAboutData(df)

def prepareXandY(df):
	feature_cols = ['opptPTS', 'teamDrtg', 'teamPF', 'teamTO', 'teamORB', 'teamFGA']
	x = df[feature_cols]
	y = df['opptRslt']
	x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.4, random_state=2)
	return x_train, x_test, y_train, y_test


def randomForest(df):
	x_train, x_test, y_train, y_test = prepareXandY(df)
	model = RandomForestClassifier()
	model.fit(x_train,y_train)
	y_predict = model.predict(x_test)
	print('accuracy of RandomForest :')
	print(accuracy_score(y_test, y_predict))

randomForest(df)

def KNNeighbors(df):
	x_train, x_test, y_train, y_test = prepareXandY(df)
	print('accuracy of KNN :')
	for k in range(10,100,10):
		model = KNeighborsClassifier(n_neighbors = k)
		model.fit(x_train,y_train)
		predictions = model.predict(x_test)
		scores = (accuracy_score(y_test,predictions))
		print('k = ',k)
		print(scores)

KNNeighbors(df)

def linearSVC(df):
	x_train, x_test, y_train, y_test = prepareXandY(df)
	clf = LinearSVC(random_state=2)
	clf.fit(x_train, y_train)
	pred = (clf.predict(x_test))
	print('accuracy of SVM :')
	print(accuracy_score(y_test, pred))
 
linearSVC(df)

def gradientBoostingClasifier(df):
	x_train, x_test, y_train, y_test = prepareXandY(df)
	clf = GradientBoostingClassifier(random_state=2)
	clf.fit(x_train, y_train)
	pred = (clf.predict(x_test))
	print('accuracy of GradientBoosting :')
	print(accuracy_score(y_test, pred))
 
gradientBoostingClasifier(df)

'''
	* df : donnée dans lesquelles on va chercher les données
	* loc -> localité : 1 = team, 2 = oppt
	* type : nom de la colonne où l'on va calculer la moyenne 
	* name : nom de l'équipe
'''
def moyenne(df, loc, type, name):
	feature_cols = ['opptPTS','opptAbbr','teamAbbr','teamDrtg', 'teamPF', 'teamTO', 'teamORB', 'teamFGA']
	tab = df[feature_cols]
	total = 0
	cpt = 0
	if (loc == 1):
		team = 'teamAbbr'
	else:
		team = 'opptAbbr'
	for i in range (0,len(tab)):
		if(df[feature_cols].at[i,team] == name ):
			cpt = cpt+1
			total = total+int(df[feature_cols].at[i,type])
	return total/cpt


def gameToPredict(df, team1, team2):
	opptPTS = moyenne(df, 2, 'opptPTS', team2)
	teamDrtg = moyenne(df, 1, 'teamDrtg', team1)
	teamPF = moyenne(df, 1, 'teamPF', team1)
	teamTO = moyenne(df, 1, 'teamTO', team1)
	teamORB = moyenne(df, 1, 'teamORB', team1)
	teamFGA = moyenne(df, 1, 'teamFGA', team1)
	tab = [[opptPTS, teamDrtg, teamPF, teamTO, teamORB, teamFGA]]
	return tab

def predictRandomForest(df,team1,team2):
	x_train, x_test, y_train, y_test = prepareXandY(df)
	clf = RandomForestClassifier()
	clf.fit(x_train, y_train)
	filename = 'nba_pred_modelv1.sav'
	pickle.dump(clf, open(filename, 'wb'))
	nba_pred_modelv1 = pickle.load(open(filename, 'rb'))
	
	g1 = gameToPredict(df, team1, team2)
	pred = nba_pred_modelv1.predict(g1)
	prob = nba_pred_modelv1.predict_proba(g1)
	if(pred[0]=='Win'):
		print(2)
	else:
		print(1)
	print(prob)

def predictGradientTreeBoosting(df,team1,team2):
	x_train, x_test, y_train, y_test = prepareXandY(df)
	clf = GradientBoostingClassifier()
	clf.fit(x_train, y_train)
	filename = 'nba_pred_modelv1.sav'
	pickle.dump(clf, open(filename, 'wb'))
	nba_pred_modelv1 = pickle.load(open(filename, 'rb'))
	
	g1 = gameToPredict(df, team1, team2)
	pred = nba_pred_modelv1.predict(g1)
	prob = nba_pred_modelv1.predict_proba(g1)
	if(pred[0]=='Win'):
		print(2)
	else:
		print(1)
	print(prob)
	
def predictKNN(df,team1,team2):
	x_train, x_test, y_train, y_test = prepareXandY(df)
	clf = KNeighborsClassifier(60)
	clf.fit(x_train, y_train)
	filename = 'nba_pred_modelv1.sav'
	pickle.dump(clf, open(filename, 'wb'))
	nba_pred_modelv1 = pickle.load(open(filename, 'rb'))
	
	g1 = gameToPredict(df, team1, team2)
	pred = nba_pred_modelv1.predict(g1)
	prob = nba_pred_modelv1.predict_proba(g1)
	if(pred[0]=='Win'):
		print(2)
	else:
		print(1)
	print(prob)
	
def getResultMatch(df, team1, team2):
	feature_cols = ['teamAbbr','teamRslt', 'opptAbbr']
	tab = df[feature_cols]
	result = 0
	for i in range (0,len(tab)):
		if(df[feature_cols].at[i,'teamAbbr'] == team1 and df[feature_cols].at[i,'opptAbbr'] == team2):
			if(df[feature_cols].at[i,'teamRslt'] == 'Win'):
				result = 1
			else:
				result = 2
			break
	print(result)
	print('--------------------')

	'''
	Les prédictions ci dessous mettent un temps enorme à s'effectuer, nous avons donc donné leurs résultats
	sur le pdf (20 matchs différents avec 3 modèles)
	'''
	
'''
predictRandomForest(df,'PHO', 'SA')
predictGradientTreeBoosting(df,'PHO', 'SA')
predictKNN(df,'PHO', 'SA')
print('Match PHO vs SA')
getResultMatch(df2,'PHO', 'SA')
  
predictRandomForest(df,'PHI','MIA')
predictGradientTreeBoosting(df,'PHI', 'MIA')
predictKNN(df,'PHI', 'MIA')
print('Match PHI vs MIA')
getResultMatch(df2,'PHI','MIA')
 
predictRandomForest(df,'CLE','BOS')
predictGradientTreeBoosting(df,'CLE','BOS')
predictKNN(df,'CLE','BOS')
print('Match CLE vs BOS')
getResultMatch(df2,'CLE','BOS')
 
predictRandomForest(df,'LAL','LAC')
predictGradientTreeBoosting(df,'LAL','LAC')
predictKNN(df,'LAL','LAC')
print('Match LAL vs LAC')
getResultMatch(df2,'LAL','LAC')
  
predictRandomForest(df,'GS','MIN')
predictGradientTreeBoosting(df,'GS','MIN')
predictKNN(df,'GS','MIN')
print('Match GS vs MIN')
getResultMatch(df2,'GS','MIN')
  
predictRandomForest(df,'UTA','SAC')
predictGradientTreeBoosting(df,'UTA','SAC')
predictKNN(df,'UTA','SAC')
print('Match UTA vs SAC')
getResultMatch(df2,'UTA','SAC')
   
predictRandomForest(df,'WAS','DET')
predictGradientTreeBoosting(df,'WAS','DET')
predictKNN(df,'WAS','DET')
print('Match WAS vs DET')
getResultMatch(df2,'WAS','DET')
  
predictRandomForest(df,'POR','MIL')
predictGradientTreeBoosting(df,'POR','MIL')
predictKNN(df,'POR','MIL')
print('Match POR vs MIL')
getResultMatch(df2,'POR','MIL')
  
predictRandomForest(df,'IND','MEM')
predictGradientTreeBoosting(df,'IND','MEM')
predictKNN(df,'IND','MEM')
print('Match IND vs MEM')
getResultMatch(df2,'IND','MEM')
  
predictRandomForest(df,'TOR','NO')
predictGradientTreeBoosting(df,'TOR','NO')
predictKNN(df,'TOR','NO')
print('Match TOR vs NO')
getResultMatch(df2,'TOR','NO')
  
predictRandomForest(df,'CHI', 'SA')
predictGradientTreeBoosting(df,'CHI', 'SA')
predictKNN(df,'CHI', 'SA')
print('Match CHI vs SA')
getResultMatch(df2,'CHI', 'SA')
  
predictRandomForest(df,'DAL','POR')
predictGradientTreeBoosting(df,'DAL','POR')
predictKNN(df,'DAL','POR')
print('Match DAL vs POR')
getResultMatch(df2,'DAL','POR')
  
predictRandomForest(df,'OKC','HOU')
predictGradientTreeBoosting(df,'OKC','HOU')
predictKNN(df,'OKC','HOU')
print('Match OKC vs HOU')
getResultMatch(df2,'OKC','HOU')
  
predictRandomForest(df,'DEN','LAC')
predictGradientTreeBoosting(df,'DEN','LAC')
predictKNN(df,'DEN','LAC')
print('Match DEN vs LAC')
getResultMatch(df2,'DEN','LAC')
  
predictRandomForest(df,'ATL','NY')
predictGradientTreeBoosting(df,'ATL','NY')
predictKNN(df,'ATL','NY')
print('Match ATL vs NY')
getResultMatch(df2,'ATL','NY')
 
predictRandomForest(df,'BKN','NY')
predictGradientTreeBoosting(df,'BKN','NY')
predictKNN(df,'BKN','NY')
print('Match BKN vs NY')
getResultMatch(df2,'BKN','NY')
  
predictRandomForest(df,'MIA','MEM')
predictGradientTreeBoosting(df,'MIA','MEM')
predictKNN(df,'MIA','MEM')
print('Match MIA vs MEM')
getResultMatch(df2,'MIA','MEM')
  
predictRandomForest(df,'GS','OKC')
predictGradientTreeBoosting(df,'GS','OKC')
predictKNN(df,'GS','OKC')
print('Match GS vs OKC')
getResultMatch(df2,'GS','OKC')
  
predictRandomForest(df,'IND','CHI')
predictGradientTreeBoosting(df,'IND','CHI')
predictKNN(df,'IND','CHI')
print('Match IND vs CHI')
getResultMatch(df2,'IND','CHI')
   
predictRandomForest(df,'CHA','DAL')
predictGradientTreeBoosting(df,'CHA','DAL')
predictKNN(df,'CHA','DAL')
print('Match CHA vs DAL')
getResultMatch(df2,'CHA','DAL')'''
