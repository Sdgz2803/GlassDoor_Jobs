import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

filePath = r"C:\Users\sdgz0\Desktop\Data Analyst Portfolio\Glassdoor Job posting\Uncleaned_DS_jobs.csv"

df = pd.read_csv(filePath)

#Remove the Index column
df.drop('index',axis=1,inplace=True)

print(df.shape)
print(df.dtypes)

# Remove Duplicates rows
df.drop_duplicates(inplace=True)

print(df.shape)

#Cleaning the company Name
df['Company Name'] = df['Company Name'].apply(lambda x: x.split('\n')[0])

# Remove Job postings without salary number
df = df[df['Salary Estimate'] != '-1']

# Simplify the dataset to glassdoor est. only
df['glassdoor est'] = df['Salary Estimate'].apply(lambda x: 1 if 'glassdoor est.' in x.lower() else 0)
df = df[df['glassdoor est'] == 1]   

#Feature engineering on the salary column 
salary = df["Salary Estimate"].apply(lambda x: x.split('(')[0])
minus_Kd = salary.apply(lambda x: x.replace('K','').replace('$',''))
df['min_salary'] = minus_Kd.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = minus_Kd.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.min_salary+df.max_salary)/2
df = df.drop(['Salary Estimate','glassdoor est'],axis=1)

#Extracting job_state from Location Column and replacing Full Names with appropriate state code and for Remote and Utah deleting the row

df['Location'].apply(lambda x: x.split(",")[-1]).value_counts()

df['job_state'] = df['Location'].apply(lambda x: x.split(",")[-1])
df['job_state'].replace(['United States','Texas','California','New Jersey','Remote','Utah'],["US","TX","CA","NJ",np.nan,np.nan],inplace=True)
df.dropna(axis=0,inplace=True)
df.reset_index(inplace=True)  

print(df['job_state'].value_counts())

#Comparing job_state and Headquaerters location

df['Headquarters1'] = df['Headquarters'].apply(lambda x: x.split(",")[-1].strip())
df['same_state'] = np.where(df['Headquarters1']==df['job_state'],1,0)
df.drop(['Headquarters1'],axis=1,inplace=True)

df.Rating = np.where(df.Rating ==-1.0,0,df.Rating)
df.sort_values(by='Rating',ascending=False)

#extracting key skills mentioned in the job description

df['Job Description'][0].split('\n\n')

df['python'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
df['excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
df['hadoop'] = df['Job Description'].apply(lambda x: 1 if 'hadoop' in x.lower() else 0)
df['spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
df['aws'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
df['sql'] = df['Job Description'].apply(lambda x: 1 if 'sql' in x.lower() else 0)
df['tableau'] = df['Job Description'].apply(lambda x: 1 if 'tableau' in x.lower() else 0)
df['big_data'] = df['Job Description'].apply(lambda x: 1 if 'big data' in x.lower() else 0)

print(df['Job Title'].value_counts()[:100])

def seniority(job_title):
    job_title=job_title.lower()
    snr=['sr','senior','lead','principal','vp','vice president','director']
    for i in snr:
        if i in job_title:
            return "senior"
    if "jr" in job_title:
        return "junior"
    return "na"

df['Seniority'] = df['Job Title'].apply(seniority)

#Simplifying Job Title
def title_simplifier(title):
    mapping = {
        'data scientist': 'data scientist',
        'data engineer': 'data engineer',
        'analyst': 'analyst',
        'machine learning': 'mle',
        'manager': 'manager',
        'director': 'director',
        'vice president': 'vp'
    }
    
    title_lower = title.lower()
    for key, value in mapping.items():
        if key in title_lower:
            return value
    return 'na'

df['job_simp'] = df['Job Title'].apply(title_simplifier)

df.drop(['index','Founded','Competitors'],axis=1,inplace=True)

df.to_csv("Cleaned_DS_jobs.csv",index=False)