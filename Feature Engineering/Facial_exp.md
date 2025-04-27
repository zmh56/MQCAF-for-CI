### Facial Expression Features

#### Feature Extraction
- **AU Units**: Action Units (AUs) represent facial muscle movements. The relevant AUs for this system include:
  - AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r, AU10_r, AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r, AU25_r, AU26_r, AU45_r.
  
These features are computed for each frame of the video to capture dynamic facial expressions over time. 

#### AU Representation
- **AU Frequency (AUP)**: The frequency of occurrence of each AU, which is defined as:
  - \( \text{AUP} = \frac{\text{Number of AU Occurrences}}{\text{Total Frames}} \)
  
- **AU Intensity (AUI)**: The average intensity of each AU, which is defined as:
  - \( \text{AUI} = \frac{\text{Total AU Intensity}}{\text{Total Frames}} \)

---

#### Emotion Frequency Analysis
- **Frequency**: The frequency of each emotion displayed during the task (e.g., anger, disgust, fear, happiness, sadness, surprise, neutral).
- **Average Duration**: The average duration of each emotional expression.
- **Maximum Duration**: The longest duration for each emotional expression during the task.


#### Emotion Metrics (Table 5.1)
| Statistic Name     | Description                                                              |
|--------------------|--------------------------------------------------------------------------|
| **Frequency (N)**  | Total occurrences of the emotion during the task.                        |
| **Frequency (N/min)** | Number of times the emotion occurred per minute.                       |
| **Total Duration (s)** | The total duration the emotion lasted during the task.                 |
| **Min Duration (s)**  | The shortest duration the emotion lasted.                              |
| **Max Duration (s)**  | The longest duration the emotion lasted.                               |
| **Average Duration (s)** | The average duration of the emotion during the task.                 |

---

### Machine Learning Feature Extraction
For reference, an example GitHub repository for facial expression feature extraction, including AU tracking and emotion classification, is available at [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace).