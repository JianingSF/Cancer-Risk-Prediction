import pandas as pd
import numpy as np
import argparse

from scipy.stats import skew
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import SelectFromModel

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

pd.set_option('display.max_columns', None)

def dataPreprocessing4Patient(df):
	#transpose and rename column names for patients
	df= df.transpose().iloc[2:] #remove the first two lines with titles

	df['track_name'] = df.index # add track_name into columns

	df.columns=['samples','profiled','profiledAlter',
	'cancerStage', 'cancerType', 'diagonosisAge', 'informed', 'histologicType', 
	'survivalDays', 'grade', 'race', 'survivalStatus', 'sex', 'vitalStatus', 'mutation','track_name']

	#drop columns with same values
	df.drop('cancerType', axis=1, inplace=True)
	df.drop('histologicType', axis=1, inplace=True)
	df.drop('samples', axis=1, inplace=True)
	
	df.drop('profiledAlter', axis=1, inplace=True)
	df.drop('survivalStatus', axis=1, inplace=True)
	
	df['diagonosisAge'] = df['diagonosisAge'].astype(float)
	df['survivalDays'] = df['survivalDays'].astype(float)
	df['mutation'] = df['mutation'].astype(float)

	df['profiled'] = df['profiled'].astype('category')
	df['cancerStage'] = df['cancerStage'].astype('category')
	df['informed'] = df['informed'].astype('category')
	df['grade'] = df['grade'].astype('category')
	df['race'] = df['race'].astype('category')
	df['sex'] = df['sex'].astype('category')
	df['vitalStatus'] = df['vitalStatus'].astype('category')

	df['track_name'] = df['track_name'].astype(str)
	
	#drop rows with None 
	df = df.dropna(subset=['grade'])
	df = df.dropna(subset=['race'])
	df = df.dropna(subset=['vitalStatus'])
	
	#add No for missing profiled
	df['profiled'] = df['profiled'].cat.add_categories('No')
	df['profiled'].fillna('No', inplace =True) 
	
	#handle upper case for 'sex'
	df['sex'] = df['sex'].mask(df['sex'] == 'MALE', 'Male')

	return df

def scaler(column):
	scaler = StandardScaler()
	x = column.values.reshape(-1, 1)
	x_scaled = scaler.fit_transform(x)
	return x_scaled
	
def robustScaler(column):
	scaler = RobustScaler()
	x = column.values.reshape(-1, 1)
	x_scaled = scaler.fit_transform(x)
	return x_scaled
	
	
def scaler4df(df):
	scaler = StandardScaler()
	trackName = df['track_name']

	df.drop('track_name', axis=1, inplace=True)
	
	scaled_features = StandardScaler().fit_transform(df.values)
	
	#To keep original indices and column names
	scaled_features_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
	
	df = pd.concat([scaled_features_df, trackName], axis=1)
	
	return df

def oneHotEncoder(column):
	ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
	x = column.values.reshape(-1, 1)
	r= ohe.fit_transform(x)
	return pd.DataFrame(r)

def normaliza4Patient(df):
	#can not use boxcom, since all must be positive values
	#df['diagonosisAge'] = boxcox(df['diagonosisAge'], 0)
	#df['survivalDays'] = boxcox(df['survivalDays'], 0)
	#df['mutation'] = boxcox(df['mutation'], 0)
	
	
	df['diagonosisAge'] = scaler(df['diagonosisAge'])
	df['survivalDays'] = scaler(df['survivalDays'])
	#df['mutation'] = scaler(df['mutation'])
	df['mutation'] = robustScaler(df['mutation'])
	
	
	df['profiled'] = LabelEncoder().fit_transform(df['profiled'])
	df['cancerStage'] = LabelEncoder().fit_transform(df['cancerStage'])
	df['informed'] = LabelEncoder().fit_transform(df['informed'])
	df['grade'] = LabelEncoder().fit_transform(df['grade'])
	df['race'] = LabelEncoder().fit_transform(df['race'])
	df['sex'] = LabelEncoder().fit_transform(df['sex'])
	df['vitalStatus'] = LabelEncoder().fit_transform(df['vitalStatus'])
	
	return df

def dataPreprocessing4MRNA(df):
#remane and remove char from SAMPLE_ID, then will match with track_name in patients
	df.rename(columns={'SAMPLE_ID':'track_name'}, inplace=True)
	df['track_name'] = df['track_name'].str[:10]
	df.drop('STUDY_ID', axis=1, inplace=True)
	df=df.dropna(axis=1,how='all')  
	df.dropna()
	
	return df

def PCA4Mrna(df4Mrna, varianceRatio):
	PCA4Mrna = df4Mrna.copy()
	PCA4Mrna.drop('track_name', axis=1, inplace=True)
	pca = PCA(varianceRatio)
	PCA4Mrna = pca.fit_transform(PCA4Mrna)

	dfPCA4Mrna = pd.DataFrame.from_records(PCA4Mrna)
	df4Mrna = pd.concat([dfPCA4Mrna, df4Mrna['track_name']], axis=1)
	return df4Mrna

def plot4Mrna(df):
	cor = df.iloc[:,0:80].corr(method ='pearson')
	sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
	plt.show()
	
def plot4Patient(df):
	plt.subplot(2,4,1)
	plt.hist(df['grade'],5)
	plt.xlabel('grade')
	  
	plt.subplot(2,4,2)
	plt.hist(df['sex'],2)
	plt.xlabel('sex') 
	
	plt.subplot(2,4,3)
	plt.hist(df['cancerStage'],15)
	plt.xlabel('cancerStage')
	
	plt.subplot(2,4,4)
	plt.hist(df['race'],3)
	plt.xlabel('race')
	
	plt.subplot(2,4,5)
	plt.hist(df['vitalStatus'],2)
	plt.xlabel('vitalStatus')
	
	plt.subplot(2,4,6)
	plt.hist(df['survivalDays'],100)
	plt.xlabel('survivalDays')
	
	plt.subplot(2,4,7)
	plt.hist(df['mutation'],100)
	plt.xlabel('mutation')
	
	plt.subplot(2,4,8)
	plt.hist(df['diagonosisAge'],100)
	plt.xlabel('diagonosisAge')
	
	plt.show()

def barPlot(df, category, value, plt):
	names = list(df[category].unique())
	x=[]
	for name in names:
		x.append(list(df[df[category]==name][value]))
	plt.hist(x, bins = 10, label=names)
	plt.legend(fontsize = 'x-small')
	plt.set_xlabel(value)
	plt.set_ylabel('Count')
	return plt
	
def plot4PatientStageGrade(df, type):
	plt.subplot(3,2,1)
	sns.boxplot(x=type, y="diagonosisAge", data=df)
	
	f = plt.subplot(3,2,2)
	f = barPlot(df, type, 'diagonosisAge', f)
	
	plt.subplot(3,2,3)
	sns.boxplot(x=type, y="mutation", data=df)
	
	f = plt.subplot(3,2,4)
	f = barPlot(df, type, 'mutation', f)
	
	plt.subplot(3,2,5)
	sns.countplot(x=type, hue="race", data=df)
	
	plt.subplot(3,2,6)
	sns.countplot(x=type, hue="sex", data=df)
	
	plt.show()
	
def plot4PatientSurvival(df):
	#sns.FacetGrid(df4Patient, hue='grade').map(plt.scatter, "survivalDays", "mutation").add_legend()
	#plt.show()
	f = plt.subplot(3,2,1)
	f = barPlot(df, 'grade', 'survivalDays', f)
	
	f = plt.subplot(3,2,2)
	f = barPlot(df, 'cancerStage', 'survivalDays', f)
	
	f = plt.subplot(3,2,3)
	f = barPlot(df, 'race', 'survivalDays', f)
	
	f = plt.subplot(3,2,4)
	f = barPlot(df, 'sex', 'survivalDays', f)
	
	plt.show()
	
	sns.FacetGrid(df, hue='grade').map(plt.scatter, "survivalDays", "diagonosisAge").add_legend()
	plt.show()

	sns.FacetGrid(df, hue='grade').map(plt.scatter, "survivalDays", "mutation").add_legend()
	plt.show()

def plot4PatientVitalStatus(df):
	
	plt.subplot(4,2,1)
	sns.countplot(x='vitalStatus', hue='race', data=df)
	
	plt.subplot(4,2,2)
	sns.countplot(x='vitalStatus', hue='sex', data=df)
	
	plt.subplot(4,2,3)
	sns.countplot(x='vitalStatus', hue='grade', data=df)
	
	plt.subplot(4,2,4)
	sns.countplot(x='vitalStatus', hue='cancerStage', data=df)
	
	f = plt.subplot(4,2,5)
	f = barPlot(df, 'vitalStatus', 'diagonosisAge', f)

	f = plt.subplot(4,2,6)
	f = barPlot(df, 'vitalStatus', 'mutation', f)
	
	f = plt.subplot(4,2,7)
	f = barPlot(df, 'vitalStatus', 'survivalDays', f)
	
	plt.show()

# Feature selection by GenericUnivariateSelect for cancerStage for numeric features
def plot4featureMutualInfo(x, y):
	mutual_information = mutual_info_classif(x, y)
	plt.subplots(1, figsize=(50, 1))
	sns.heatmap(mutual_information[:, np.newaxis].T, cmap='Blues', cbar=False, linewidths=5, annot=True)
	#plt.yticks([], [])
	plt.gca().set_xticklabels(x.columns[::], rotation=45, ha='right', fontsize=12)
	plt.suptitle("Variable Importance for Cancer Stage", fontsize=18, y=1.2)
	plt.gcf().subplots_adjust(wspace=0.2)
	plt.show()
	
def featureSelectionByStatistics4CancerStage(df4Mrna,trainingData):
	cols = [col for col in df4Mrna.columns]
	x = trainingData[cols].copy()
	x.drop('track_name', axis=1, inplace=True)
	x = pd.concat([x[:], trainingData['diagonosisAge'], trainingData['mutation']], axis=1)
	y = trainingData['cancerStage']
	
	#plot
	plot4featureMutualInfo(x,y)
	
	trans = GenericUnivariateSelect(score_func=mutual_info_classif, mode='percentile', param=50)
	xTrans = trans.fit_transform(x, y)
	columnsSelected = x.columns[trans.get_support()].values
	print ('columnsSelected by Statistics for cancer stage: ', 'Total numbers is ', len(columnsSelected), 'Columns are ', columnsSelected)

# Feature selection from model for cancerStage for numeric features
def featureSelectionFromModel(patient, mrna, trainingData, type):
	cols = [col for col in mrna.columns]
	x = trainingData[cols].copy()
	x.drop('track_name', axis=1, inplace=True)
	if (type == 'cancerStage' or type == 'grade' or type == 'survivalDays'):
		x = pd.concat([x[:], trainingData['diagonosisAge'], trainingData['mutation']], axis=1)
	if (type == 'vitalStatus'):
		x = pd.concat([x[:], trainingData['diagonosisAge'], trainingData['mutation'], trainingData['survivalDays']], axis=1)
	
	y = trainingData[type]
	
	#feature selection from model
	if (type == 'survivalDays'):
		clf = DecisionTreeRegressor()
	else:
		clf = DecisionTreeClassifier()
	trans = SelectFromModel(clf, threshold='median') #0.01
	xTrans = trans.fit_transform(x, y)
	
	columnsSelected = x.columns[trans.get_support()].values
	print ('Selected features from model for ', type, ': Total numbers is ', len(columnsSelected),  '\nFeatures are: \n', columnsSelected)
	
	#Plot feature_importances_
	clf.fit(x, y)
	plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
	plt.xticks(range(len(clf.feature_importances_)), x.columns, rotation=270)
	plt.title('Feature Importance for ' + type)
	plt.show()
	
	dfSelectedFeatures = pd.DataFrame(xTrans, columns=columnsSelected)
	dfSelectedFeatures4Merge = pd.concat([dfSelectedFeatures[:], trainingData['track_name']], axis=1)
	
	otherColumns = patient.columns.difference(dfSelectedFeatures.columns)
	trainingData = pd.merge(dfSelectedFeatures4Merge, patient[otherColumns], on='track_name', how='inner')
	return trainingData
	
#OneHotEncoding
def feature4StageAndGrade(df):
	dfProfiled = oneHotEncoder(df['profiled'])
	dfInformed = oneHotEncoder(df['informed'])
	dfRace = oneHotEncoder(df['race'])
	dfSex = oneHotEncoder(df['sex'])
	
	columns_to_drop = ['profiled', 'informed','race','sex','cancerStage', 'grade','vitalStatus', 'track_name', 'survivalDays' ]
	df.drop(labels=columns_to_drop, axis=1, inplace=True)
	
	x = pd.concat([df[:], dfProfiled, dfInformed, dfRace, dfSex], axis=1)
	return x

def feature4VitalStatus(df):
	#Keep cancerStage, grade and survivalDays as feature
	dfProfiled = oneHotEncoder(df['profiled'])
	dfInformed = oneHotEncoder(df['informed'])
	dfRace = oneHotEncoder(df['race'])
	dfSex = oneHotEncoder(df['sex'])
	dfCancerStage = oneHotEncoder(df['cancerStage'])
	dfGrade = oneHotEncoder(df['grade'])
	
	columns_to_drop = ['profiled', 'informed','race','sex','cancerStage', 'grade','vitalStatus', 'track_name']
	df.drop(labels=columns_to_drop, axis=1, inplace=True)
	
	x = pd.concat([df[:], dfProfiled, dfInformed, dfRace, dfSex, dfCancerStage, dfGrade], axis=1)
	return x

def feature4SurvivalDays(df):
	#Keep cancerStage, grade as feature
	dfProfiled = oneHotEncoder(df['profiled'])
	dfInformed = oneHotEncoder(df['informed'])
	dfRace = oneHotEncoder(df['race'])
	dfSex = oneHotEncoder(df['sex'])
	dfCancerStage = oneHotEncoder(df['cancerStage'])
	dfGrade = oneHotEncoder(df['grade'])
	
	columns_to_drop = ['profiled', 'informed','race','sex','cancerStage', 'grade','vitalStatus', 'survivalDays', 'track_name']
	df.drop(labels=columns_to_drop, axis=1, inplace=True)
	
	x = pd.concat([df[:], dfProfiled, dfInformed, dfRace, dfSex, dfCancerStage, dfGrade], axis=1)
	return x
	
#########Prediction for CancerStage, Grade, VitalStatus, Survival Days
def predictCancerStage(df):
	#print (trainingData['cancerStage'].value_counts())
	y = df['cancerStage']
	x = feature4StageAndGrade(df)
	
	totalEstimators = [10,20,30,40,50]
	for n_estimator in totalEstimators:
		model = RandomForestClassifier(n_estimators=n_estimator, max_depth=3, random_state=0)
		scores = cross_val_score(model, x, y, scoring='accuracy', cv=2)
		print("Model parameter of n_estimator is : ", n_estimator)
		print ('Predict accuracy of cancer stage', scores.mean())
		
		if n_estimator == totalEstimators[-1]:
			model.fit(x,y)
			predictions = model.predict(x)
			print ('Confusion matrix for CancerStage:')
			print (confusion_matrix(y, predictions))

def predictGrade(df):
	#print (trainingData['cancerStage'].value_counts())
	y = df['grade']
	x = feature4StageAndGrade(df)
	
	totalEstimators = [10,20,30,40,50]
	for n_estimator in totalEstimators:
		model = RandomForestClassifier(n_estimators=n_estimator, max_depth=3, random_state=0)
		scores = cross_val_score(model, x, y, scoring='accuracy', cv=2)
		print("Model parameter of n_estimator is : ", n_estimator)
		print ('Predict accuracy of grade', scores.mean())
		if n_estimator == totalEstimators[-1]:
			model.fit(x,y)
			predictions = model.predict(x)
			print ('Confusion matrix for Grade:')
			print (confusion_matrix(y, predictions))

def predictVitalStatus(df):
	y = df['vitalStatus']
	x = feature4VitalStatus(df)
	
	totalEstimators = [10,20,30,40,50]
	for n_estimator in totalEstimators:
		model = RandomForestClassifier(n_estimators=n_estimator, max_depth=3, random_state=0)
		scores = cross_val_score(model, x, y, scoring='accuracy', cv=2)
		print("Model parameter of n_estimator is : ", n_estimator)
		print ('Predict accuracy of vital status', scores.mean())
		
		if n_estimator == totalEstimators[-1]:
			model.fit(x,y)
			predictions = model.predict(x)
			print ('Confusion matrix for VitalStatus:')
			print (confusion_matrix(y, predictions))
		
def predictSurvivalDays(df):
	y = df['survivalDays']
	x = feature4SurvivalDays(df)
	
	totalEstimators = [10,20,30,40,50]
	for n_estimator in totalEstimators:
		model = RandomForestRegressor(n_estimators=n_estimator, max_depth=3, random_state=0)
		scores = cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=2)
		#neg_mean_squared_error, neg_mean_absolute_error
		print("Model parameter of n_estimator is : ", n_estimator)
		print ('Predict MSE of Survival Days', scores.mean())

def parseCmd():
    parser = argparse.ArgumentParser(description='Risk Prediction')
    parser.add_argument("--rootPath", type=str, default="cancerStage", help="File root path")
    parser.add_argument("--predictType", type=str, default="cancerStage", help="What to predict") 
    parser.add_argument("--ifSelectFeature", type=str, default="yes", help="If select feature from model")
    return parser.parse_args()

def main():
	opts = parseCmd()
	
	#import data
	#rootPath="/Users/tang_li/Desktop/CG/"

	patientPath=opts.rootPath+ "patient_data.tsv"
	df4Patient = pd.read_table(patientPath)

	mrnaPath=opts.rootPath+ "mrna_data.txt"
	df4Mrna = pd.read_table(mrnaPath)
	
	seqPath=opts.rootPath+ "seq_data.txt"
	df4Seq = pd.read_table(seqPath)
	df4Seq=df4Seq.dropna(axis=1,how='all')
	#print (df4Seq.describe())

	###########Patient pre-processing

	df4Patient = dataPreprocessing4Patient(df4Patient)
	
	######PLOT for patient info relationship
	#plot4Patient(df4Patient)
	#plot4PatientStageGrade(df4Patient, 'cancerStage')
	#plot4PatientStageGrade(df4Patient, 'grade')
	#plot4PatientSurvival(df4Patient)
	#plot4PatientVitalStatus(df4Patient)
   	
	######Scale and encode category into interger
	df4Patient = normaliza4Patient(df4Patient)
	
	###########Mrna pre-processing 
	df4Mrna = dataPreprocessing4MRNA(df4Mrna)
	#####Scale features by z-score
	df4Mrna = scaler4df(df4Mrna)
	#plot4Mrna(df4Mrna)
	
	#PCA dimension reduction for Mrna
	varianceRatio = 0.95  #to control features
	#df4Mrna = PCA4Mrna(df4Mrna, varianceRatio)
	#print ('df4Mrna', df4Mrna.shape)

	##############################
	#Prepare training data for modeling 
	trainingData = pd.merge(df4Patient, df4Mrna, on='track_name', how='inner')

	############Feature selection
	#Feature Selection By Statistics  (for compare with From model)
	#trainingData = featureSelectionByStatistics4CancerStage (df4Mrna,trainingData)
	
	#feature Selection from model
	if (opts.ifSelectFeature == 'yes'):
		trainingData = featureSelectionFromModel(df4Patient, df4Mrna,trainingData, opts.predictType)
	
	print ('Feature selection: ' + opts.ifSelectFeature)
	
	#########Modeling to predict CancerStage, Grade, VitalStatus, Survival Days
	if (opts.predictType == 'cancerStage'):
		#predict cancer stage
		trainingData4CancerStage = trainingData.copy()
		predictCancerStage(trainingData4CancerStage)
	
	if (opts.predictType == 'grade'):	
		trainingData4Grade = trainingData.copy()
		predictGrade(trainingData4Grade)
	
	if (opts.predictType == 'survivalDays'):
		trainingData4SurvivalDays = trainingData.copy()
		predictSurvivalDays(trainingData4SurvivalDays)
		
	if (opts.predictType == 'vitalStatus'):
		trainingData4VitalStatus = trainingData.copy()
		predictVitalStatus(trainingData4VitalStatus)


	print (df4Patient.shape)
	print (df4Mrna.shape)
	print (trainingData.shape)

if __name__ == "__main__":
    main()