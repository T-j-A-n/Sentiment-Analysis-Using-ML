## MECE Audience Segmentation for Retention  

---

## 📄 Overview  
This project implements a hybrid **unsupervised + supervised machine learning pipeline** for **customer retention and engagement segmentation**.  
It combines **K-Means and Gaussian Mixture Models (GMM)** for audience clustering and uses **ensemble models (Random Forest, Gradient Boosting)** for predictive retention analysis.

The goal is to achieve **MECE (Mutually Exclusive, Collectively Exhaustive)** audience segmentation — identifying distinct, actionable user groups for targeted retention strategies.

---

## 🧩 Project Workflow  

### 1️⃣ Data Preprocessing
- Loaded dataset: `user_personalized_features.csv`
- Handled missing values and encoded categorical features (`Gender`, `Location`, `Interests`, `Newsletter_Subscription`).
- Standardized numerical features using `StandardScaler`.
- Engineered additional behavioral variables:
  - `engagement_score` → weighted metric combining engagement features.
  - `purchased` → derived binary target (high-value vs. low-value).
  - `visitor` → encoded as New (0) or Returning (1).
  - `bought_count` → simulated purchase count feature.

---

### 2️⃣ Exploratory Data Analysis (EDA)
Visual analysis performed using **Seaborn** and **Matplotlib**:
- Feature distributions, correlations, and missing values heatmap.
- Scatterplots of engagement vs. spending.
- Boxplots and heatmaps by visitor type, gender, and interest.
- Correlation matrix showing key relationships between engagement, time spent, and spending.

---

### 3️⃣ Feature Engineering Summary
| Feature | Description |
|----------|--------------|
| `engagement_score` | Weighted sum of pages viewed, time spent, and purchase frequency |
| `purchased` | Binary label based on engagement and spending thresholds |
| `visitor` | Categorical (0 = New, 1 = Returning) |
| `bought_count` | Synthetic behavioral feature for demonstration |

---

### 4️⃣ Unsupervised Learning – Clustering

#### 🌀 K-Means Clustering
- Used scaled behavioral features: `Total_Spending`, `Pages_Viewed`, `Time_Spent_on_Site_Minutes`, `Purchase_Frequency`, `Average_Order_Value`, and `engagement_score`.
- Determined optimal **k** using the **Elbow Method** and **Silhouette Score**.
- Achieved:
  - **Silhouette Score:** `0.33`
  - **Mean Inter-Cluster Distance:** `2.46`
- Indicates moderately distinct, meaningful segmentation.

#### 🎯 Gaussian Mixture Model (GMM)
- Applied to capture **soft cluster assignments** (users can belong to multiple clusters probabilistically).
- Better suited for non-spherical, overlapping behavioral segments.

#### ⚗️ K-Means + GMM Ensemble
- Combined outputs of both models through **majority voting ensemble**.
- Evaluated with clustering quality metrics:
  - Silhouette Score
  - Davies–Bouldin Index
  - Calinski–Harabasz Index
- PCA visualization used for 2D projection of ensemble clusters.

---

### 5️⃣ Cluster Quality Analysis

| Metric | Value | Interpretation |
|---------|--------|----------------|
| Silhouette Score | 0.33 | Moderate separation – realistic in behavioral datasets |
| Mean Inter-Cluster Distance | 2.46 | Reasonable distance between centroids |
| WCSS | (Computed) | Compact clusters, minimal internal spread |

**Conclusion:**  
Clusters are distinct enough to represent real behavioral groups while capturing overlap between user types.

---

### 6️⃣ Cluster Profiles

| Cluster | Behavior | Description |
|----------|-----------|-------------|
| 0 | High engagement, high spending | Loyal / VIP Users |
| 1 | Moderate activity, consistent behavior | Potential Loyalists |
| 2 | Low engagement, low spending | At-Risk / New Users |

✅ These clusters are **actionable** and **interpretable**, supporting data-driven retention strategies.

---

### 7️⃣ Supervised Learning – Predictive Modeling

To complement clustering, supervised models were trained on the engineered `purchased` target variable:

#### Models:
- Support Vector Machine (SVM)
- Random Forest Classifier
- Gradient Boosting Classifier

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|---------|-----------|
| SVM | ~0.80 | 0.78 | 0.75 | 0.76 |
| **Random Forest** | **~0.88** | **0.87** | **0.86** | **0.86** |
| Gradient Boosting | ~0.85 | 0.84 | 0.83 | 0.84 |

**Best Model:**  
✅ **Random Forest Ensemble** — robust, interpretable, and most accurate for predicting purchase/retention likelihood.

Feature importance showed that:
- `Total_Spending`
- `Pages_Viewed`
- `Time_Spent_on_Site_Minutes`
were key predictors.

---

### 8️⃣ Evaluation Metrics Summary

| Metric | Description | Ideal Trend |
|---------|--------------|-------------|
| Silhouette Score | Cluster compactness and separation | Higher = Better |
| Davies–Bouldin Index | Cluster overlap ratio | Lower = Better |
| Calinski–Harabasz Index | Between/within variance ratio | Higher = Better |
| Accuracy / F1 | Predictive performance | Higher = Better |

---

### 9️⃣ Insights & Business Recommendations

1. **Customer Segments**
   - Cluster 0 (Loyal): Prioritize loyalty programs and rewards.  
   - Cluster 1 (Potential): Target with personalized offers to boost engagement.  
   - Cluster 2 (At-Risk): Re-engage through reactivation campaigns.

2. **Predictive Modeling**
   - Random Forest can predict which users are likely to purchase again.
   - Combine with clustering for smarter campaign targeting.

3. **Strategic Takeaway**
   - Hybrid segmentation (K-Means + GMM) yields interpretable, stable clusters.
   - Ensemble learning enhances predictive accuracy for retention outcomes.

---

### 🔮 Future Work
- Explore **DBSCAN** or **Hierarchical Clustering** for irregular cluster structures.  
- Test **HDBSCAN** for automatic cluster count estimation.  
- Fine-tune **GMM covariance types** for better soft membership modeling.  
- Integrate this segmentation into a **real-time recommendation system**.

---

## 🧰 Tech Stack

| Category | Tools |
|-----------|--------|
| Data | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Scaling | `StandardScaler` |
| Clustering | `KMeans`, `GaussianMixture`, `DBSCAN` |
| Evaluation | `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score` |
| Classification | `RandomForestClassifier`, `SVC`, `GradientBoostingClassifier` |
| Dimensionality Reduction | `PCA` |

---

## 🚀 Key Takeaways
- **K-Means + GMM Ensemble** provides balanced, interpretable audience segmentation.  
- **Random Forest** delivers the highest retention prediction accuracy.  
- A **hybrid approach** (segmentation + prediction) leads to actionable retention strategies.  
- Moderate silhouette and good inter-cluster distance confirm meaningful, realistic segmentation.  

---

## 🧠 Author Notes
This project demonstrates the synergy between **unsupervised clustering** for segmentation and **supervised learning** for prediction in the context of **AI-driven customer retention**.  
It adheres to MECE segmentation principles — clusters are distinct (mutually exclusive) yet cover all customer types (collectively exhaustive).

---

