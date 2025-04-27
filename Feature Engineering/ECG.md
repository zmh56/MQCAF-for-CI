### Feature Description

The following is a classification list based on time-domain, frequency-domain, non-linear, entropy measures, heart rate asymmetry, heart rate fragmentation, and fractal & complexity features:

### Time-Domain Features
Time-domain features are directly obtained from the time series of NN intervals (i.e., time intervals between heartbeats).

| Feature Parameter | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| ECG_Rate_Mean     | The average number of heartbeats within a given time period, usually expressed in beats per minute (bpm). |
| HRV_MeanNN        | The mean NN interval, reflecting the average heart rate.                     |
| HRV_SDNN          | The standard deviation of NN intervals, reflecting overall heart rate variability. |
| HRV_RMSSD         | The root mean square of successive differences between adjacent NN intervals, reflecting short-term heart rate variability. |
| HRV_SDSD          | The standard deviation of successive differences between adjacent NN intervals, also reflecting short-term heart rate variability. |
| HRV_CVNN          | The coefficient of variation of NN intervals (CVNN=SDNN/MeanNN), reflecting relative variability. |
| HRV_CVSD          | The ratio of RMSSD to mean NN interval, another measure of relative variability. |
| HRV_MedianNN      | The median of NN intervals, reflecting the median level of heart rate.        |
| HRV_MadNN         | The median absolute deviation of NN intervals, reflecting consistency in heart rate. |
| HRV_MCVNN         | The coefficient of variation of the median.                                   |
| HRV_IQRNN         | The interquartile range of NN intervals, reflecting the dispersion of heart rate distribution. |
| HRV_SDRMSSD       | The ratio of SDNN to RMSSD, reflecting different aspects of heart rate variability. |
| HRV_Prc20NN       | The 20th percentile of NN intervals.                                         |
| HRV_Prc80NN       | The 80th percentile of NN intervals.                                         |
| HRV_pNN50         | The percentage of NN intervals with differences greater than 50ms, reflecting large heart rate variations. |
| HRV_pNN20         | The percentage of NN intervals with differences greater than 20ms, reflecting small heart rate variations. |
| HRV_MinNN         | The minimum NN interval, i.e., the shortest heartbeat interval.               |
| HRV_MaxNN         | The maximum NN interval, i.e., the longest heartbeat interval.               |
| HRV_HTI           | The heart rate triangle index, derived from analyzing the histogram of NN intervals, reflecting overall heart rate variability. |
| HRV_TINN          | The width of the triangular interpolation of the NN interval histogram, another geometric measure of heart rate variability. |

### Frequency-Domain Features
Frequency-domain features are analyzed by converting the time series into the frequency domain, revealing heart rate variability at different frequency components.

| Feature Parameter | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| HRV_LF            | Power in the low-frequency range, reflecting the balance of sympathetic and parasympathetic activity. |
| HRV_HF            | Power in the high-frequency range, mainly reflecting parasympathetic activity. |
| HRV_VHF           | Power in the very high-frequency range, whose physiological significance is not fully understood. |
| HRV_TP            | Total power, reflecting the overall heart rate variability across all frequencies. |
| HRV_LFHF          | The ratio of LF to HF, reflecting the balance of sympathetic and parasympathetic activity. |
| HRV_LFn           | Normalized low-frequency power, LF/(TP-VHF), adjusted for the influence of VHF. |
| HRV_HFn           | Normalized high-frequency power, HF/(TP-VHF), similarly adjusted for the influence of VHF. |
| HRV_LnHF          | The natural logarithm of high-frequency power, used to reduce data skewness. |

### Non-linear Features
Non-linear features describe the dynamic characteristics of heart rate signals through more complex mathematical models, reflecting non-linear dynamic behavior in heart rate variability.

| Feature Parameter | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| HRV_SD1           | The standard deviation in the direction perpendicular to the line of identity in the Poincaré plot, reflecting short-term heart rate variability. |
| HRV_SD2           | The standard deviation in the direction along the line of identity in the Poincaré plot, reflecting long-term heart rate variability. |
| HRV_SD1SD2        | The ratio of SD1 to SD2, reflecting the balance between short-term and long-term heart rate variability. |
| HRV_S             | The area of the Poincaré plot, reflecting the overall dynamics of heart rate variability. |
| HRV_CSI           | The complexity index, SD2/SD1, reflecting the complexity of heart rate variability. |
| HRV_CVI           | The complexity variability index, the logarithmic product of SD1 and SD2, reflecting the complexity and irregularity of the heart rate sequence. |
| HRV_CSI_Modified  | A modified complexity index, calculated based on a different approach. |
| HRV_DFA_alpha2    | The alpha2 coefficient in fractal dimension analysis, reflecting the self-similarity of long-term heart rate variability. |
| HRV_MFDFA_alpha2_Width | The width of the multifractal spectrum, reflecting the multifractal characteristics of heart rate signals. |
| HRV_MFDFA_alpha2_Peak  | The peak value in the multifractal spectrum, indicating the strongest multifractal characteristic of the heart rate signal. |
| HRV_MFDFA_alpha2_Mean  | The mean value in the multifractal spectrum, reflecting the overall multifractal characteristics. |
| HRV_MFDFA_alpha2_Max   | The maximum value in the multifractal spectrum, indicating the maximum multifractal characteristic. |
| HRV_MFDFA_alpha2_Delta | The delta value in the multifractal spectrum, reflecting the diversity of heart rate signals. |
| HRV_MFDFA_alpha2_Asymmetry | The asymmetry in the multifractal spectrum, reflecting the directional behavior of heart rate variability. |
| HRV_MFDFA_alpha2_Fluctuation | The fluctuation in the multifractal spectrum, reflecting the stability of heart rate signals. |
| HRV_MFDFA_alpha2_Increment | The increment in the multifractal spectrum, indicating the trend of heart rate variability. |

### Entropy Measure Features
Entropy measure features use the concept of entropy to assess the uncertainty and complexity of heart rate time series.

| Feature Parameter | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| HRV_ApEn          | Approximate entropy, measuring the novelty and unpredictability of heart rate time series. |
| HRV_sampEn        | Sample entropy, similar to approximate entropy but less sensitive to data length, used to measure the complexity of the heart rate sequence. |
| HRV_shanEn        | Shannon entropy, measuring the information content of heart rate time series. |
| HRV_FuzzyEn       | Fuzzy entropy, an entropy measure using fuzzy logic to improve the calculation, reflecting the complexity of heart rate sequences. |
| HRV_MSEn          | Multi-scale entropy, analyzing the complexity of signals across different time scales. |
| HRV_CMSEn         | Composite multi-scale entropy, an improvement to multi-scale entropy, increasing sensitivity to heart rate variability analysis. |
| HRV_RCMSEn        | Refined composite multi-scale entropy, further improving the calculation of multi-scale entropy. |

### Heart Rate Asymmetry Features (HRA)
Heart rate asymmetry features describe the asymmetry in heart rate variability, revealing differences between heart rate acceleration and deceleration.

| Feature Parameter | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| HRV_GI            | Guzik index, reflecting the asymmetry of NN intervals in the Poincaré plot between heart rate acceleration and deceleration. |
| HRV_SI            | Slope index, describing the degree of slope in the Poincaré plot shape, reflecting asymmetry in heart rate variability. |
| HRV_AI            | Area index, evaluating the asymmetry in heart rate variability by calculating the area of specific regions in the Poincaré plot. |
| HRV_PI            | Porta index, measuring the ratio of increasing to decreasing RR intervals in the Poincaré plot, reflecting asymmetry in heart rate variability. |
| HRV_C1d           | Contribution of heart rate deceleration to short-term heart rate variability. |
| HRV_C1a           | Contribution of heart rate acceleration to short-term heart rate variability. |
| HRV_SD1d          | Short-term standard deviation of heart rate deceleration.                  |
| HRV_SD1a          | Short-term standard deviation of heart rate acceleration.                  |
| HRV_C2d           | Contribution of heart rate deceleration to long-term heart rate variability. |
| HRV_C2a           | Contribution of heart rate acceleration to long-term heart rate variability. |
| HRV_SD2d          | Long-term standard deviation of heart rate deceleration.                   |
| HRV_SD2a          | Long-term standard deviation of heart rate acceleration.                   |
| HRV_Cd            | Contribution of heart rate deceleration to total heart rate variability.   |
| HRV_Ca            | Contribution of heart rate acceleration to total heart rate variability.   |
| HRV_SDNNd         | Total standard deviation of heart rate deceleration.                       |
| HRV_SDNNa         | Total standard deviation of heart rate acceleration.                       |

### Heart Rate Fragmentation Features (HRF)
Heart rate fragmentation features reflect discontinuous or irregular patterns in the heart rate time series, which may be related to physiological or pathological states.

| Feature Parameter | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| HRV_PIP           | The percentage of inflection points in the NN interval sequence, reflecting irregularity in heart rate variability. |
| HRV_IALS          | The inverse of the average length of acceleration/deceleration segments in the NN interval sequence, reflecting the degree of fragmentation. |
| HRV_PSS           | The percentage of short segments, reflecting the proportion of transient patterns in the NN interval sequence. |
| HRV_PAS           | The percentage of alternating segments, reflecting the proportion of alternating patterns in the NN interval sequence. |

### Fractal & Complexity Features
Fractal and complexity features describe the complex structure and self-similarity of heart rate variability, reflecting multi-level regulation mechanisms in the physiological system.

| Feature Parameter | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| HRV_CD            | Correlation dimension, reflecting the fractal characteristics of the heart rate time series. |
| HRV_HFD           | Higuchi fractal dimension, measuring the complexity and self-similarity of the heart rate time series. |
| HRV_KFD           | Katz fractal dimension, another method for measuring the complexity of heart rate time series. |
| HRV_LZC           | Lempel-Ziv complexity, reflecting the complexity and pattern diversity in heart rate time series. |

---

### GitHub ECG Feature Extraction Library
https://github.com/chandanacharya1/ECG-Feature-extraction-using-Python
https://github.com/tkhan11/Time-Series-Feature-Extraction-ECG
https://github.com/Seb-Good/ecg-features
