import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set style for all plots
plt.style.use('seaborn')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12

def create_student_metrics_visualization():
    """Create visualization comparing student vs non-student metrics"""
    # Create comparison data
    metrics = {
        'Segment': ['Student', 'Student', 'Non-Student', 'Non-Student'],
        'Metric': ['Late Payment Rate', 'Avg Tenancy (months)', 'Late Payment Rate', 'Avg Tenancy (months)'],
        'Value': [0.68, 4, 0.32, 6.8]  # Based on the analysis
    }
    df = pd.DataFrame(metrics)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Segment', y='Value', hue='Metric', data=df)
    plt.title('Student vs Non-Student Storage Metrics')
    plt.ylabel('Value')
    plt.savefig('student_metrics.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_revenue_by_size_visualization():
    """Create visualization of revenue per square foot by unit size"""
    # Create unit size data
    sizes = [15, 25, 50, 75, 100, 150]
    revenue_multiplier = [1, 1.2, 1.8, 2.1, 2.3, 2.3]
    
    plt.figure(figsize=(12, 6))
    plt.plot(sizes, revenue_multiplier, marker='o', linewidth=2)
    plt.title('Revenue Multiplier by Unit Size')
    plt.xlabel('Unit Size (sq ft)')
    plt.ylabel('Revenue Multiplier (vs 15 sq ft)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('revenue_by_size.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_seasonal_patterns():
    """Create visualization of seasonal demand patterns"""
    # Create monthly data
    months = range(1, 13)
    student_demand = [0.8, 0.7, 0.9, 1.0, 1.2, 1.8, 1.9, 1.7, 1.0, 0.8, 0.9, 1.5]
    regular_demand = [1.0, 0.9, 1.1, 1.2, 1.1, 1.0, 1.0, 1.0, 0.9, 0.9, 1.0, 1.1]
    
    plt.figure(figsize=(12, 6))
    plt.plot(months, student_demand, marker='o', label='Student Demand', linewidth=2)
    plt.plot(months, regular_demand, marker='s', label='Regular Demand', linewidth=2)
    plt.title('Seasonal Demand Patterns')
    plt.xlabel('Month')
    plt.ylabel('Demand Index (1.0 = Average)')
    plt.xticks(months, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('seasonal_patterns.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_hourly_patterns():
    """Create visualization of hourly visit patterns with service transition"""
    # Create hourly data
    hours = range(9, 19)  # 9 AM to 6 PM
    visits = [5, 8, 10, 15, 18, 12, 8, 5, 3, 2]  # Sample visit patterns
    
    plt.figure(figsize=(12, 6))
    
    # Plot visit pattern
    plt.plot(hours, visits, marker='o', linewidth=2, label='Visit Pattern')
    
    # Add service transition visualization
    plt.axvspan(9, 17, color='lightblue', alpha=0.3, label='Reception Hours (9:00-17:00)')
    plt.axvspan(17, 18, color='lightgreen', alpha=0.3, label='Online Customer Service (17:00-18:00)')
    
    plt.title('Daily Visit Pattern and Service Coverage')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Number of Visits')
    plt.xticks(hours, [f'{h:02d}:00' for h in hours])
    
    # Add annotations
    plt.annotate('Reception\nCloses', xy=(17, 3), xytext=(17, 10),
                arrowprops=dict(facecolor='black', shrink=0.05),
                ha='center', va='bottom')
    plt.annotate('Online Service\nOnly', xy=(17.5, 2), xytext=(17.5, 8),
                ha='center', va='bottom')
    
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('hourly_patterns.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_implementation_timeline():
    """Create visualization of implementation timeline"""
    # Create timeline data
    tasks = ['Adjust Hours', 'Unit Consolidation', 'Student Redirection', 'Marketing Campaign', 'Monitoring']
    start_months = [1, 2, 4, 4, 1]
    durations = [1, 4, 3, 6, 12]
    
    plt.figure(figsize=(12, 6))
    for i, (task, start, duration) in enumerate(zip(tasks, start_months, durations)):
        plt.barh(i, duration, left=start, height=0.3)
        plt.text(start+0.1, i, task)
    
    plt.title('Implementation Timeline')
    plt.xlabel('Month')
    plt.yticks([])
    plt.xlim(0, 13)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('implementation_timeline.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_financial_impact():
    """Create visualization of projected financial impact"""
    # Create impact data
    categories = ['Space Optimization', 'Customer Targeting', 'Operating Hours', 'Total Impact']
    revenue_impact = [17.5, 12.5, 6, 22]  # Midpoint of ranges
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(categories, revenue_impact)
    plt.title('Projected Financial Impact')
    plt.ylabel('Percentage Improvement (%)')
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}%', ha='center', va='bottom')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('financial_impact.png', bbox_inches='tight', dpi=300)
    plt.close()

def generate_customer_data(n_samples=1000):
    """Generate synthetic customer data for visualization"""
    np.random.seed(42)
    
    # Generate base data
    data = {
        'customer_id': range(1, n_samples + 1),
        'age_group': np.random.choice(['18-22', '23-30', '31-40', '41-50', '51+'], n_samples,
                                    p=[0.25, 0.2, 0.25, 0.2, 0.1]),
        'tenancy_duration': np.random.normal(6, 2, n_samples),  # months
        'payment_reliability': np.random.normal(0.85, 0.1, n_samples),
        'unit_size': np.random.choice([15, 25, 50, 75, 100, 150], n_samples),
        'monthly_rate': np.random.normal(100, 30, n_samples),
        'enquiry_frequency': np.random.poisson(3, n_samples),  # enquiries per month
        'enquiry_channel': np.random.choice(['in_store', 'phone', 'website'], n_samples, 
                                          p=[0.3, 0.3, 0.4]),
        'contract_renewals': np.random.poisson(1, n_samples),
        'conversion_rate': np.random.normal(0.7, 0.1, n_samples),  # 70% baseline conversion
        'enquiry_month': np.random.randint(1, 13, n_samples),  # Month of enquiry
        'questionnaire_student': np.random.choice([1, 0, -1], n_samples, p=[0.3, 0.5, 0.2])  # 1=Yes, 0=No, -1=Not Answered
    }
    
    df = pd.DataFrame(data)
    
    # Define summer months (May to September)
    summer_months = [5, 6, 7, 8, 9]
    
    # Determine student status based on questionnaire and fallback logic
    def determine_student_status(row):
        # If questionnaire is answered, use that
        if row['questionnaire_student'] != -1:
            return row['questionnaire_student']
        
        # Fallback logic for unanswered questionnaires
        is_student_age = row['age_group'] in ['18-22', '23-30']
        is_summer = row['enquiry_month'] in summer_months
        is_short_term = row['tenancy_duration'] <= 2
        
        # Apply fallback logic for student classification
        if is_student_age and is_summer and is_short_term:
            return 1
        return 0
    
    # Apply student status determination
    df['is_student'] = df.apply(determine_student_status, axis=1)
    
    # Adjust values based on student status
    for i, row in df.iterrows():
        if row['is_student']:
            df.at[i, 'tenancy_duration'] *= 0.59  # 41% shorter
            df.at[i, 'payment_reliability'] *= 0.68  # 32% lower reliability
            df.at[i, 'unit_size'] = np.random.choice([15, 25], 1)[0]  # Smaller units
            df.at[i, 'enquiry_frequency'] *= 1.2  # More frequent enquiries
            df.at[i, 'conversion_rate'] *= 1.1  # 10% higher conversion rate
            # Students more likely to use website
            if np.random.random() < 0.6:
                df.at[i, 'enquiry_channel'] = 'website'
    
    return df

def create_customer_clustering_visualization(df):
    """Create visualization of customer clusters"""
    # Prepare data for clustering
    features = ['tenancy_duration', 'payment_reliability', 'monthly_rate', 'enquiry_frequency']
    X = df[features].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create multiple visualizations
    fig = plt.figure(figsize=(20, 15))
    
    # 1. PCA Cluster Plot
    ax1 = fig.add_subplot(221)
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis')
    ax1.set_title('Customer Segments (PCA)')
    ax1.set_xlabel('First Principal Component')
    ax1.set_ylabel('Second Principal Component')
    plt.colorbar(scatter, ax=ax1, label='Cluster')
    
    # 2. Feature Relationships by Cluster
    ax2 = fig.add_subplot(222)
    scatter = ax2.scatter(df['tenancy_duration'], df['payment_reliability'], 
                         c=df['cluster'], cmap='viridis', alpha=0.6)
    ax2.set_title('Tenancy Duration vs Payment Reliability')
    ax2.set_xlabel('Tenancy Duration (months)')
    ax2.set_ylabel('Payment Reliability Score')
    plt.colorbar(scatter, ax=ax2, label='Cluster')
    
    # 3. Cluster Characteristics
    ax3 = fig.add_subplot(223)
    cluster_means = df.groupby('cluster')[features].mean()
    cluster_means_scaled = (cluster_means - cluster_means.mean()) / cluster_means.std()
    cluster_means_scaled.plot(kind='bar', ax=ax3)
    ax3.set_title('Cluster Characteristics')
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Standardized Value')
    plt.xticks(rotation=45)
    
    # 4. Unit Size Distribution by Cluster
    ax4 = fig.add_subplot(224)
    for cluster in range(4):
        cluster_data = df[df['cluster'] == cluster]['unit_size']
        ax4.hist(cluster_data, bins=20, alpha=0.5, label=f'Cluster {cluster}')
    ax4.set_title('Unit Size Distribution by Cluster')
    ax4.set_xlabel('Unit Size (sq ft)')
    ax4.set_ylabel('Count')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('customer_clustering.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_student_behavior_visualization(df):
    """Create detailed visualization of student vs non-student behavior patterns"""
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Payment Reliability Distribution
    ax1 = fig.add_subplot(221)
    student_data = df[df['is_student'] == 1]['payment_reliability']
    non_student_data = df[df['is_student'] == 0]['payment_reliability']
    ax1.hist([student_data, non_student_data], bins=20, label=['Student', 'Non-Student'], alpha=0.5)
    ax1.set_title('Payment Reliability Distribution')
    ax1.set_xlabel('Payment Reliability Score')
    ax1.set_ylabel('Count')
    ax1.legend()
    
    # 2. Unit Size Distribution
    ax2 = fig.add_subplot(222)
    student_sizes = df[df['is_student'] == 1]['unit_size']
    non_student_sizes = df[df['is_student'] == 0]['unit_size']
    ax2.boxplot([student_sizes, non_student_sizes], labels=['Student', 'Non-Student'])
    ax2.set_title('Unit Size Distribution')
    ax2.set_ylabel('Unit Size (sq ft)')
    
    # 3. Enquiry Frequency vs Tenancy Duration
    ax3 = fig.add_subplot(223)
    scatter = ax3.scatter(df['tenancy_duration'], df['enquiry_frequency'], 
                         c=df['is_student'], cmap='coolwarm', alpha=0.6)
    ax3.set_title('Enquiry Frequency vs Tenancy Duration')
    ax3.set_xlabel('Tenancy Duration (months)')
    ax3.set_ylabel('Enquiries per Month')
    plt.colorbar(scatter, ax=ax3, label='Student Status')
    
    # 4. Enquiry Channel Distribution
    ax4 = fig.add_subplot(224)
    channel_data = pd.crosstab(df['is_student'], df['enquiry_channel'], normalize='index') * 100
    channel_data.plot(kind='bar', ax=ax4)
    ax4.set_title('Enquiry Channel Distribution')
    ax4.set_xlabel('Student Status')
    ax4.set_ylabel('Percentage of Enquiries')
    ax4.legend(title='Channel')
    plt.xticks(rotation=45)
    
    # Add percentage labels on bars
    for container in ax4.containers:
        ax4.bar_label(container, fmt='%.1f%%')
    
    plt.tight_layout()
    plt.savefig('student_behavior_patterns.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_customer_lifecycle_visualization(df):
    """Create visualization of customer lifecycle patterns"""
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Customer Value Matrix
    ax1 = fig.add_subplot(221)
    scatter = ax1.scatter(df['monthly_rate'], df['tenancy_duration'], 
                         c=df['payment_reliability'], cmap='viridis', 
                         s=100*df['contract_renewals']+50, alpha=0.6)
    ax1.set_title('Customer Value Matrix')
    ax1.set_xlabel('Monthly Rate ($)')
    ax1.set_ylabel('Tenancy Duration (months)')
    plt.colorbar(scatter, ax=ax1, label='Payment Reliability')
    
    # 2. Retention Analysis
    ax2 = fig.add_subplot(222)
    retention_data = pd.crosstab(df['is_student'], df['contract_renewals'])
    retention_data.plot(kind='bar', ax=ax2)
    ax2.set_title('Customer Retention Analysis')
    ax2.set_xlabel('Student Status')
    ax2.set_ylabel('Number of Customers')
    ax2.legend(title='Number of Renewals')
    plt.xticks(rotation=45)
    
    # 3. Value Segments
    ax3 = fig.add_subplot(223)
    df['total_value'] = df['monthly_rate'] * df['tenancy_duration']
    df['value_segment'] = pd.qcut(df['total_value'], q=4, labels=['Low', 'Medium', 'High', 'Premium'])
    value_dist = df.groupby(['is_student', 'value_segment']).size().unstack()
    value_dist.plot(kind='bar', ax=ax3)
    ax3.set_title('Value Segment Distribution')
    ax3.set_xlabel('Student Status')
    ax3.set_ylabel('Number of Customers')
    plt.xticks(rotation=45)
    
    # 4. Enquiry Pattern Analysis
    ax4 = fig.add_subplot(224)
    
    # Calculate mean enquiries by channel and student status
    channel_means = df.groupby(['is_student', 'enquiry_channel'])['enquiry_frequency'].mean().unstack()
    
    # Create grouped bar plot
    channel_means.plot(kind='bar', ax=ax4)
    ax4.set_title('Enquiry Patterns by Channel')
    ax4.set_xlabel('Student Status')
    ax4.set_ylabel('Average Enquiries per Month')
    ax4.legend(title='Channel')
    
    # Add value labels on bars
    for container in ax4.containers:
        ax4.bar_label(container, fmt='%.1f')
    
    plt.tight_layout()
    plt.savefig('customer_lifecycle.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_enquiry_channel_analysis(df):
    """Create detailed visualization of enquiry channels"""
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Channel Distribution Over Time
    ax1 = fig.add_subplot(221)
    channel_counts = df['enquiry_channel'].value_counts()
    ax1.pie(channel_counts, labels=channel_counts.index, autopct='%1.1f%%',
            colors=sns.color_palette("Set2"))
    ax1.set_title('Overall Enquiry Channel Distribution')
    
    # 2. Channel Preference by Student Status
    ax2 = fig.add_subplot(222)
    channel_student = pd.crosstab(df['enquiry_channel'], df['is_student'], normalize='columns') * 100
    channel_student.plot(kind='bar', ax=ax2)
    ax2.set_title('Channel Preference by Student Status')
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Percentage of Users')
    ax2.legend(['Non-Student', 'Student'])
    plt.xticks(rotation=45)
    
    # 3. Enquiry Frequency by Channel
    ax3 = fig.add_subplot(223)
    sns.boxplot(data=df, x='enquiry_channel', y='enquiry_frequency', ax=ax3)
    ax3.set_title('Enquiry Frequency Distribution by Channel')
    ax3.set_xlabel('Channel')
    ax3.set_ylabel('Enquiries per Month')
    
    # 4. Channel Success Rate (using payment reliability as proxy)
    ax4 = fig.add_subplot(224)
    channel_success = df.groupby('enquiry_channel')['payment_reliability'].mean()
    channel_success.plot(kind='bar', ax=ax4)
    ax4.set_title('Channel Success Rate')
    ax4.set_xlabel('Channel')
    ax4.set_ylabel('Average Payment Reliability')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for container in ax4.containers:
        ax4.bar_label(container, fmt='%.2f')
    
    plt.tight_layout()
    plt.savefig('enquiry_channel_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_segment_discovery_visualization(df):
    """Create visualization showing how we discovered the student segment's importance"""
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Age Group Analysis
    ax1 = fig.add_subplot(221)
    age_conversion = df.groupby('age_group')['conversion_rate'].mean().sort_values(ascending=False)
    age_conversion.plot(kind='bar', ax=ax1)
    ax1.set_title('Conversion Rate by Age Group')
    ax1.set_xlabel('Age Group')
    ax1.set_ylabel('Average Conversion Rate')
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, v in enumerate(age_conversion):
        ax1.text(i, v, f'{v:.1%}', ha='center', va='bottom')
    
    # 2. Unit Size Preference by Age
    ax2 = fig.add_subplot(222)
    sns.boxplot(data=df, x='age_group', y='unit_size', ax=ax2)
    ax2.set_title('Unit Size Distribution by Age Group')
    ax2.set_xlabel('Age Group')
    ax2.set_ylabel('Unit Size (sq ft)')
    plt.xticks(rotation=45)
    
    # 3. Payment Reliability Patterns
    ax3 = fig.add_subplot(223)
    reliability_age = df.groupby('age_group')['payment_reliability'].agg(['mean', 'std']).sort_values('mean')
    reliability_age['mean'].plot(kind='bar', yerr=reliability_age['std'], ax=ax3, capsize=5)
    ax3.set_title('Payment Reliability by Age Group')
    ax3.set_xlabel('Age Group')
    ax3.set_ylabel('Payment Reliability Score')
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, v in enumerate(reliability_age['mean']):
        ax3.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # 4. Enquiry Channel Preference
    ax4 = fig.add_subplot(224)
    channel_age = pd.crosstab(df['age_group'], df['enquiry_channel'], normalize='index') * 100
    channel_age.plot(kind='bar', stacked=True, ax=ax4)
    ax4.set_title('Enquiry Channel Preference by Age Group')
    ax4.set_xlabel('Age Group')
    ax4.set_ylabel('Percentage of Enquiries')
    ax4.legend(title='Channel', bbox_to_anchor=(1.05, 1))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('segment_discovery.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_conversion_analysis(df):
    """Create visualization of conversion patterns that led to student focus"""
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Conversion Rate by Segment
    ax1 = fig.add_subplot(221)
    conv_segment = df.groupby('is_student')['conversion_rate'].mean()
    conv_segment.plot(kind='bar', ax=ax1)
    ax1.set_title('Conversion Rate by Segment')
    ax1.set_xlabel('Student Status')
    ax1.set_ylabel('Average Conversion Rate')
    ax1.set_xticklabels(['Non-Student', 'Student'])
    
    # Add value labels
    for i, v in enumerate(conv_segment):
        ax1.text(i, v, f'{v:.1%}', ha='center', va='bottom')
    
    # 2. Conversion Rate by Channel and Segment
    ax2 = fig.add_subplot(222)
    channel_conv = df.groupby(['enquiry_channel', 'is_student'])['conversion_rate'].mean().unstack()
    channel_conv.plot(kind='bar', ax=ax2)
    ax2.set_title('Conversion Rate by Channel and Segment')
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Average Conversion Rate')
    ax2.legend(['Non-Student', 'Student'])
    plt.xticks(rotation=45)
    
    # Add value labels
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.1f%%')
    
    # 3. Unit Size vs Conversion Rate
    ax3 = fig.add_subplot(223)
    scatter = ax3.scatter(df['unit_size'], df['conversion_rate'], 
                         c=df['is_student'], cmap='coolwarm', alpha=0.6)
    ax3.set_title('Unit Size vs Conversion Rate')
    ax3.set_xlabel('Unit Size (sq ft)')
    ax3.set_ylabel('Conversion Rate')
    plt.colorbar(scatter, ax=ax3, label='Student Status')
    
    # 4. Conversion Trend by Enquiry Frequency
    ax4 = fig.add_subplot(224)
    df['enquiry_group'] = pd.qcut(df['enquiry_frequency'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    freq_conv = df.groupby(['enquiry_group', 'is_student'])['conversion_rate'].mean().unstack()
    freq_conv.plot(kind='line', marker='o', ax=ax4)
    ax4.set_title('Conversion Rate by Enquiry Frequency')
    ax4.set_xlabel('Enquiry Frequency Group')
    ax4.set_ylabel('Average Conversion Rate')
    ax4.legend(['Non-Student', 'Student'])
    
    plt.tight_layout()
    plt.savefig('conversion_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_student_classification_visualization(df):
    """Create visualization showing how students are classified through questionnaire and fallback logic"""
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Classification Source Distribution
    ax1 = fig.add_subplot(221)
    classification_source = df['questionnaire_student'].map({1: 'Questionnaire - Student',
                                                           0: 'Questionnaire - Non-Student',
                                                           -1: 'Fallback Logic'}).value_counts()
    ax1.pie(classification_source, labels=classification_source.index, autopct='%1.1f%%',
            colors=sns.color_palette("Set2"))
    ax1.set_title('Student Classification Source Distribution')
    
    # 2. Fallback Logic Success Rate
    ax2 = fig.add_subplot(222)
    fallback_data = df[df['questionnaire_student'] == -1]
    fallback_results = fallback_data['is_student'].value_counts()
    ax2.bar(['Non-Student', 'Student'], fallback_results)
    ax2.set_title('Results of Fallback Logic Classification')
    ax2.set_ylabel('Number of Customers')
    
    # Add percentage labels
    total_fallback = fallback_results.sum()
    for i, v in enumerate(fallback_results):
        percentage = v / total_fallback * 100
        ax2.text(i, v, f'{percentage:.1f}%', ha='center', va='bottom')
    
    # 3. Summer vs Non-Summer Classification (Fallback Only)
    ax3 = fig.add_subplot(223)
    summer_months = [5, 6, 7, 8, 9]
    fallback_data['is_summer'] = fallback_data['enquiry_month'].isin(summer_months)
    summer_class = pd.crosstab(fallback_data['is_summer'], fallback_data['is_student'])
    summer_class.plot(kind='bar', ax=ax3)
    ax3.set_title('Summer vs Non-Summer Classification (Fallback Logic)')
    ax3.set_xticklabels(['Non-Summer', 'Summer'])
    ax3.set_ylabel('Number of Customers')
    ax3.legend(['Non-Student', 'Student'])
    
    # 4. Age Group Distribution by Classification Method
    ax4 = fig.add_subplot(224)
    df['classification_method'] = df['questionnaire_student'].map({1: 'Questionnaire - Student',
                                                                 0: 'Questionnaire - Non-Student',
                                                                 -1: 'Fallback Logic'})
    age_class = pd.crosstab(df['age_group'], df['classification_method'])
    age_class.plot(kind='bar', ax=ax4)
    ax4.set_title('Age Group Distribution by Classification Method')
    ax4.set_xlabel('Age Group')
    ax4.set_ylabel('Number of Customers')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('student_classification.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating visualizations...")
    
    # Generate synthetic customer data
    customer_df = generate_customer_data(1000)
    
    # Create student classification visualization first
    create_student_classification_visualization(customer_df)
    
    # Create discovery visualizations
    create_segment_discovery_visualization(customer_df)
    create_conversion_analysis(customer_df)
    
    # Create original visualizations
    create_student_metrics_visualization()
    create_revenue_by_size_visualization()
    create_seasonal_patterns()
    create_hourly_patterns()
    create_implementation_timeline()
    create_financial_impact()
    
    # Create customer segmentation visualizations
    create_customer_clustering_visualization(customer_df)
    create_student_behavior_visualization(customer_df)
    create_customer_lifecycle_visualization(customer_df)
    create_enquiry_channel_analysis(customer_df)
    
    print("Visualizations created successfully.") 