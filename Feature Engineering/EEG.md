### EEG Feature Description

#### Statistical Features
- **mean**: The average value of the data.
- **std**: The standard deviation of the data, indicating the degree of fluctuation around the mean.

#### Non-linear Features
- **lyapunov_exponent**: A measure of chaos and sensitive dependence on initial conditions in dynamic systems.

#### Waveform Features
- **peak_amplitude**: The highest point of the signal waveform.

#### Band Energy Features
- **band_energy_δ**: The energy in the δ band (typically 0.5-4 Hz).
- **band_energy_θ**: The energy in the θ band (typically 4-8 Hz).
- **band_energy_α**: The energy in the α band (typically 8-12 Hz).
- **band_energy_β**: The energy in the β band (typically 12-30 Hz).
- **band_energy_γ**: The energy in the γ band (typically 30-100 Hz).

#### Spectral Features
- **spectro_entropy**: The complexity or uncertainty of the spectrum.
- **we_entropy**: The wavelet entropy, based on wavelet transform of the signal.

#### Differential Entropy Features
- **differential_entropy**: The entropy of continuous signals, used to measure the uncertainty of the signal.

#### Complexity Features
- **apen**: Approximate entropy, a method to measure the complexity of time series data.
- **pe**: Permutation entropy, a complexity measure based on the sequential relationships of data.
- **sampen**: Sample entropy, an improved version of approximate entropy, to measure the complexity of time series data.
- **fuzzy**: Fuzzy entropy, a measure of the complexity of the signal.

#### Differential Features
- **diff1_mean**: The mean of the first-order difference.
- **diff1_std**: The standard deviation of the first-order difference.
- **diff2_mean**: The mean of the second-order difference.
- **diff2_std**: The standard deviation of the second-order difference.
- **diff3_mean**: The mean of the third-order difference.
- **diff3_std**: The standard deviation of the third-order difference.

---
ref: https://github.com/sari-saba-sadiya/EEGExtract