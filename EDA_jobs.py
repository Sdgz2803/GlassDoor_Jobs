import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# File path to the dataset
file_path = r"C:\Users\sdgz0\Desktop\Data Analyst Portfolio\Glassdoor Job posting\Cleaned_DS_jobs.csv"

# Load the dataset
df = pd.read_csv(file_path)

# --- Helper Functions --- #

def standardize_job_title(title):
    """Standardize job titles based on keywords."""
    title_lower = title.lower()
    if 'data scientist' in title_lower:
        return 'Data Scientist'
    elif 'data engineer' in title_lower:
        return 'Data Engineer'
    elif 'machine learning' in title_lower or 'ml' in title_lower:
        return 'Machine Learning Engineer'
    elif 'data analyst' in title_lower or 'analytics' in title_lower:
        return 'Data Analyst'
    elif 'research scientist' in title_lower or 'scientist' in title_lower:
        return 'Scientist'
    elif 'developer' in title_lower:
        return 'Developer'
    else:
        return 'Other'

def convert_size(size):
    """Convert company size to numeric for analysis."""
    if pd.isnull(size) or size in ['-1', 'Unknown']:
        return np.nan
    elif '10000+' in size:
        return 10000
    else:
        return int(size.split(' to ')[-1].replace(' employees', '').strip())

# --- Data Preparation --- #

# Standardize job titles
df['Standardized Job Title'] = df['Job Title'].apply(standardize_job_title)

# Convert company size to numeric
df['Size_numeric'] = df['Size'].apply(convert_size)

# --- Visualizations --- #

# 1. Salary Ranges by Job Title
salary_summary = df.groupby('Standardized Job Title')['avg_salary'].agg(['min', 'max', 'mean']).reset_index()

plt.figure(figsize=(12, 8))
sns.barplot(
    data=salary_summary,
    x='Standardized Job Title',
    y='mean',
    palette='coolwarm'
)
plt.errorbar(
    x=range(len(salary_summary)),
    y=salary_summary['mean'],
    yerr=[salary_summary['mean'] - salary_summary['min'], salary_summary['max'] - salary_summary['mean']],
    fmt='o',
    color='black',
    capsize=4
)
plt.xticks(rotation=45, ha='right')
plt.title('Salary Ranges by Job Title with Min & Max', fontsize=16)
plt.xlabel('Job Title', fontsize=14)
plt.ylabel('Salary ($)', fontsize=14)
plt.tight_layout()
plt.savefig('salary_ranges_by_job_title.png')
plt.show()

# 2. Average Salary by State with Job Counts
state_salary_pivot = df.groupby('job_state').agg(
    avg_salary=('avg_salary', 'mean'),
    job_count=('job_state', 'count')
).sort_values('avg_salary', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(
    x=state_salary_pivot['avg_salary'],
    y=state_salary_pivot.index,
    palette='coolwarm'
)
for i, (avg_salary, job_count) in enumerate(zip(state_salary_pivot['avg_salary'], state_salary_pivot['job_count'])):
    plt.text(avg_salary + 2, i, f'{job_count} jobs', va='center', fontsize=10)
plt.xlabel('Average Salary ($)', fontsize=14)
plt.ylabel('State', fontsize=14)
plt.title('Average Salary by State with Job Counts', fontsize=16)
plt.tight_layout()
plt.savefig('avg_salary_by_state_with_counts.png')
plt.show()

# 3. Average Salary by Industry with Job Counts
industry_salary_pivot = df.groupby('Industry').agg(
    avg_salary=('avg_salary', 'mean'),
    job_count=('Industry', 'count')
).sort_values('avg_salary', ascending=False)

plt.figure(figsize=(14, 10))
sns.barplot(
    x=industry_salary_pivot['avg_salary'],
    y=industry_salary_pivot.index,
    palette='Blues_r'
)
for i, (avg_salary, job_count) in enumerate(zip(industry_salary_pivot['avg_salary'], industry_salary_pivot['job_count'])):
    plt.text(avg_salary + 1, i, f'{job_count} Jobs', va='center', fontsize=10, color='black')
plt.xlabel('Average Salary ($)', fontsize=14)
plt.ylabel('Industry', fontsize=14)
plt.title('Average Salary by Industry with Job Counts', fontsize=18, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('avg_salary_by_industry_with_counts.png')
plt.show()

# 4. Most In-Demand Skills
demand_by_skill = df[['python', 'excel', 'hadoop', 'spark', 'aws', 'sql', 'tableau', 'big_data']].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    y=demand_by_skill.index,
    x=demand_by_skill.values,
    palette='viridis'
)
plt.title('Most In-Demand Technical Skills', fontsize=16)
plt.xlabel('Counts of Job Listings', fontsize=12)
plt.ylabel('Technical Skill', fontsize=12)
plt.tight_layout()
plt.savefig('most_in_demand_skills.png')
plt.show()

# --- Correlation Analysis --- #

salary_rating_corr = df['avg_salary'].corr(df['Rating'])
salary_size_corr = df['avg_salary'].corr(df['Size_numeric'])

print(f"Correlation between salary and rating: {salary_rating_corr:.2f}")
print(f"Correlation between salary and size: {salary_size_corr:.2f}")

# Scatter Plot: Salary vs. Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Rating', y='avg_salary', alpha=0.6)
plt.title('Salary vs. Company Rating', fontsize=16)
plt.xlabel('Company Rating', fontsize=12)
plt.ylabel('Average Salary ($)', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('salary_vs_rating.png')
plt.show()

# Scatter Plot: Salary vs. Company Size
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Size_numeric', y='avg_salary', alpha=0.6)
plt.title('Salary vs. Company Size', fontsize=16)
plt.xlabel('Company Size (Max Employees)', fontsize=12)
plt.ylabel('Average Salary ($)', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('salary_vs_size.png')
plt.show()