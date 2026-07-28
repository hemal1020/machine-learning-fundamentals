# K-Means Clustering — Student Performance Grouping

Unsupervised clustering to find natural groupings of students based on theory marks, lab marks, and
attendance.

**Dataset:** 150 students. `Student_ID` is a plain numeric row index, not a real identifier — excluded
from modeling. No missing values.

**Approach:** *k* chosen via elbow method (WCSS) + silhouette score for each feature pairing, rather
than picked arbitrarily. Clustered on (theory, attendance), (lab, attendance), and a bonus clustering on
all 3 features together, visualized with PCA.

**Results:**

| Clustering | k chosen | Silhouette score |
|---|---|---|
| Theory vs Attendance | 3 | 0.672 |
| Lab vs Attendance | 3 | 0.681 |
| All 3 features (PCA-visualized) | 3 | 0.676 |

All three consistently resolve to the same 3-group structure: a high-performing group, a mid group, and a
low group, holding across theory, lab, and attendance together.

**What I learned:** the elbow method alone can be ambiguous to read by eye, so pairing it with silhouette
score gives an objective number to actually compare k values on. I also learned `random_state` matters for
reproducibility — without it, K-means initializes randomly and cluster labels/boundaries can shift between
runs. And that clustering with a larger k isn't automatically better — more clusters only helps if
silhouette score confirms they're genuinely better separated, not just fragmenting one real group.

**Note on the data:** `Student_ID` is a synthetic numeric index rather than any real student name, roll
number, or other identifying detail, so this dataset doesn't raise privacy concerns for public upload.
