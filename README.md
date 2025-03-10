# Student Storage Optimization: Business Analysis Project

## Project Overview
This project analyzes a self-storage business's operations with a focus on student customer segments and operational efficiency. The analysis integrates 13 years of booking data to identify opportunities for revenue optimization while maintaining service quality during a period of rising operational costs.

## Key Analysis Areas
1. **Student Customer Analysis**
   - Payment reliability patterns
   - Tenancy duration analysis
   - Seasonal demand fluctuations
   - Unit size preferences

2. **Space Utilization**
   - Revenue per square foot optimization
   - Unit size profitability analysis
   - Seasonal occupancy patterns
   - Turnover cost impact

3. **Service Hours Optimization**
   - Reception vs online service hours
   - Visit pattern analysis
   - Staff utilization
   - Digital service transition

## Key Findings
- Student segment shows 68% higher late payment rates
- 41% shorter average tenancy in student segments
- Larger units generate 2.3x more revenue per square foot
- Clear seasonal patterns in student demand (summer/December peaks)

## Project Structure
```
.
├── Analysis Documents
│   ├── presentation.md         # Executive presentation
│   └── detailed_analysis.md    # Comprehensive analysis
│
├── Visualizations
│   ├── student_metrics.png     # Student vs non-student comparisons
│   ├── revenue_by_size.png     # Unit size revenue analysis
│   ├── seasonal_patterns.png   # Demand patterns
│   ├── hourly_patterns.png     # Visit patterns and service hours
│   ├── implementation_timeline.png
│   └── financial_impact.png
│
├── Code
│   ├── create_visualizations.py # Visualization generation
│   └── requirements.txt        # Python dependencies
│
└── Documentation
    ├── README.md              # This file
    └── cursor.md              # Development environment setup
```

## Recommendations
1. **Space Optimization**
   - Consolidate 15 sqft units at premium locations
   - Convert to larger units where possible
   - Expected impact: 15-20% revenue increase

2. **Customer Segmentation**
   - Redirect student customers to unmanned sites
   - Focus premium locations on long-term customers
   - Expected impact: 10-15% cost reduction

3. **Service Hours**
   - Reception closes at 5:00 PM
   - Online customer service until 6:00 PM
   - Expected impact: 5-7% cost savings

## Implementation Timeline
- **Phase 1** (Months 1-3): Service hours adjustment
- **Phase 2** (Months 4-6): Space optimization
- **Phase 3** (Months 7-12): Monitoring and refinement

## Running the Analysis

### Prerequisites
```bash
# Install required packages
pip install -r requirements.txt
```

### Generate Visualizations
```bash
python create_visualizations.py
```

## Technologies Used
- Python: Primary analysis language
- pandas/numpy: Data manipulation
- matplotlib/seaborn: Data visualization
- SQL: Data extraction and analysis

## Expected Outcomes
- 22% increase in revenue per square foot
- Improved operational efficiency
- Enhanced customer service through digital channels
- Better alignment of resources with demand patterns

## Contributing
This project is part of a business analysis case study. For modifications or improvements, please follow standard git workflow:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License
This project is proprietary and confidential. All rights reserved.