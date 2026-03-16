import itertools
# from .prompters import LlamaVisionPrompter, MistralVisionPrompter, QwenVisionPrompter

import itertools

def build_valid_embedding_strings(embedding_types=("ust", "vis", "ts", "lets")) -> list[str]:
    valid = []
    for r in range(2, 4):  # lengths 2..3
        for combo in itertools.combinations(embedding_types, r):
            s = set(combo)
            if "ust" not in s:
                continue
            if "ts" in s and "lets" in s:
                continue
            for perm in itertools.permutations(combo):
                valid.append(",".join(perm))
    return valid



TITLE_MAPPINGS = {
    "RWC": "Right Whale Call",
    "TEE": "Transient Electromagnetic Event",
    "ECG": "ECG Heart Rhythm",
    "EMG": "Electromyogram (EMG)",
    "CTU": "24-Hour Power Consumption",
    "HAR": "Tri-Axial Accelerometer",
    "HAD": "Human Activity Recognition"
}


LABEL_MAPPING = {
    "RWC": {0: "No right whale call", 1: "Right whale call present"},
    "TEE": {
        0: "CG Positive Initial Return Stroke",
        1: "IR Negative Initial Return Stroke",
        2: "SR Subsequent Negative Return Stroke",
        3: "I Impulsive Event",
        4: "I2 Impulsive Event Pair (TIPP)",
        5: "KM Gradual Intra-Cloud Stroke",
        6: "O Off-record",
    },
    "ECG": {
        0: "Normal sinus rhythm",
        1: "Atrial fibrillation",
        2: "Other rhythm",
        3: "Too noisy to classify",
    },
    "EMG": {0: "Healthy (normal)", 1: "Myopathy", 2: "Neuropathy"},
    "CTU": {0: "Desktop", 1: "Laptop"},  # aligns with your Computers dataset
    "HAR": {
        0: "WALKING",
        1: "WALKING_UPSTAIRS",
        2: "WALKING_DOWNSTAIRS",
        3: "SITTING",
        4: "STANDING",
        5: "LAYING",
    },
    "HAD": {
        1: "Walking Forward",
        2: "Walking Left",
        3: "Walking Right",
        4: "Walking Upstairs",
        5: "Walking Downstairs",
        6: "Running Forward",
        7: "Jumping Up",
        8: "Sitting",
        9: "Standing",
        10: "Sleeping",
        11: "Elevator Up",
        12: "Elevator Down",
    },
}


X_MAPPINGS = {
    "RWC": "Time (2-second window)",
    "TEE": "Time since trigger (µs)",
    "ECG": "Time (30–60 second window)",
    # "EMG": "Time (~4-second window)",
    "EMG": "Time (ms)",

    "CTU": "Time (24-hours)",
    "HAR": "Time (2.56-second window)",
    "HAD": "Time (1.28-second window at 100Hz)"
}


Y_MAPPINGS = {
    "RWC": "Hz",
    "TEE": "VHF power density",
    "ECG": "ECG amplitude (mV)",
    "EMG": "EMG amplitude (µV)",
    "CTU": "Power Consumption (Watts)",
    "HAR": "Acceleration (g)",
    "HAD": {
        "acc": "Acceleration (g)",
        "gyro": "Angular Velocity (dps)",
    }
}


LEGEND_MAPPINGS = {
    "HAR": ["X-axis", "Y-axis", "Z-axis"],
    "CTU": None,
    "ECG": None,
    "EMG": None,
    "TEE": None,
    "RWC": None,
}
    

TASK_DESCRIPTION = {
    "CTU": (
        "Play as a computer energy consumption analysis expert: determine whether this computer is a desktop or a laptop "
        "based on the 24-hour power consumption time series."
    ),
    "ECG": "As a cardiologist, classify the patient's heart rhythm from a single-lead ECG segment.",
    "EMG": "As an EMG analysis expert, determine the subject type based on the EMG time series.",
    "HAR": (
        "As a human activity recognition expert, determine the activity based on the tri-axial accelerometer series (x, y, z) over time."
    ),
    "TEE": (
        "Based on the FORTE satellite power-density time series, select the transient electromagnetic event that best matches. "
        "There are seven event types with distinct temporal patterns (e.g., sharp turn-on + noise, slow ramp-up to attachment + spike + exponential decay, paired impulsive peaks, gradual intra-cloud increase, off-record)."
    ),
    "RWC": (
        "Play the role of a marine biology expert: decide whether the recording contains a North Atlantic right whale call "
        "(e.g., an up-call with a rising contour ~0.5-1.5 s, typically ~50-300 Hz)."
    ),
    "HAD": (
        "As a human activity recognition expert, classify the activity performed by the subject "
        "based on the 6-axis IMU data (tri-axial accelerometer and tri-axial gyroscope) collected "
        "from a wearable MotionNode sensor at 100Hz. Analyze the temporal patterns in both "
        "acceleration (in g) and angular velocity (in degrees per second) to determine which of "
        "the 12 daily activities the subject is performing."
    ),
}


HELP_STRING = {
    "CTU": "determine whether a computer is a desktop or a laptop based on its 24-hour power consumption time series.",
    "ECG": "classify a patient's heart rhythm from a single-lead ECG segment.",
    "EMG": "determine the subject type based on the EMG time series.",
    "HAR": "determine the human activity based on tri-axial accelerometer data over time.",
    "TEE": "select the transient electromagnetic event type that best matches the FORTE satellite power-density time series.",
    "RWC": "decide whether a recording contains a North Atlantic right whale call based on its audio time series.",
    "HAD": (
        "classify human activities based on 6-axis inertial measurement unit (IMU) data including "
        "tri-axial accelerometer and gyroscope readings from a wearable sensor."
    ),
}


EXTRA_INFO_MAPPINGS = {
    "RWC": "To successfully classify transient electromagnetic events based on power density time series data from the FORTE satellite, you would analyze specific features from the time series that correspond to the unique characteristics of each event type. Here are the key features and patterns to focus on for each of the seven event types: CG Positive Initial Return Stroke: Sharp Onset of Radiation: Look for a very quick rise in power density at the beginning of the event. Short Duration Noise: After the sharp onset, expect a period of noisy signal lasting a few hundred  microseconds. IR Negative Initial Return Stroke: Slow Ramp-Up: The power density will increase gradually until it reaches a specific threshold. Spike at Attachment Point: After the slow ramp-up, look for a significant spike in the power density. Exponential Decline: Post-spike, the waveform should show an exponentially shaped decline. SR Subsequent Negative Return Stroke: Multiple Peaks: Since these strokes occur after the initial return strokes, identify multiple peaks that might follow initial peaks in a given timeframe. Characteristics Similar to IR Negative: Each subsequent return stroke might mirror the slow ramp-up and sharp spike, though typically less pronounced than the initial stroke. I Impulsive Event: Sudden Peak: Look for a sudden, sharp peak in the power density without prior gradual increase or subsequent pairs of peaks. I2 Impulsive Event Pair: Paired Peaks: Identify closely spaced pairs of sharp peaks. Consistency in Time Interval: The time interval between the paired peaks should be consistent across events classified as TIPPs. KM Gradual Intra-Cloud Stroke: Gradual Increase in Power: Unlike impulsive events, these will show a more gradual rise in power density. Sustained High Power Levels: The power might stay elevated for a longer period compared to other intra-cloud events. O Off-record: Incomplete Waveform: Look for waveforms that seem to cut off or end abruptly without resolving normally within the 800-microsecond timeframe. To automate the classification process and improve accuracy, you can implement a series of steps: Preprocessing: Apply noise reduction and normalization techniques to clean the data for more precise analysis. Feature Extraction: Develop algorithms to extract the above features from the time series data. This might include detecting peaks, analyzing the rate of rise and fall in power density, and measuring durations and intervals. Classification Model: Use machine learning techniques such as decision trees, support vector machines, or neural networks to classify events based on the extracted features. Training the model with labeled examples of each event type will be crucial. Validation and Testing: Continuously validate the model with new data and adjust parameters to handle variations in signal characteristics or noise levels. By focusing on these features and employing robust data processing and machine learning techniques, you can effectively classify the types of transient electromagnetic events detected by the FORTE satellite",
    "TEE": "To successfully classify transient electromagnetic events based on power density time series data from the FORTE satellite, you would analyze specific features from the time series that correspond to the unique characteristics of each event type. Here are the key features and patterns to focus on for each of the seven event types: CG Positive Initial Return Stroke: Sharp Onset of Radiation: Look for a very quick rise in power density at the beginning of the event. Short Duration Noise: After the sharp onset, expect a period of noisy signal lasting a few hundred  microseconds. IR Negative Initial Return Stroke: Slow Ramp-Up: The power density will increase gradually until it reaches a specific threshold. Spike at Attachment Point: After the slow ramp-up, look for a significant spike in the power density. Exponential Decline: Post-spike, the waveform should show an exponentially shaped decline. SR Subsequent Negative Return Stroke: Multiple Peaks: Since these strokes occur after the initial return strokes, identify multiple peaks that might follow initial peaks in a given timeframe. Characteristics Similar to IR Negative: Each subsequent return stroke might mirror the slow ramp-up and sharp spike, though typically less pronounced than the initial stroke. I Impulsive Event: Sudden Peak: Look for a sudden, sharp peak in the power density without prior gradual increase or subsequent pairs of peaks. I2 Impulsive Event Pair: Paired Peaks: Identify closely spaced pairs of sharp peaks. Consistency in Time Interval: The time interval between the paired peaks should be consistent across events classified as TIPPs. KM Gradual Intra-Cloud Stroke: Gradual Increase in Power: Unlike impulsive events, these will show a more gradual rise in power density. Sustained High Power Levels: The power might stay elevated for a longer period compared to other intra-cloud events. O Off-record: Incomplete Waveform: Look for waveforms that seem to cut off or end abruptly without resolving normally within the 800-microsecond timeframe. To automate the classification process and improve accuracy, you can implement a series of steps: Preprocessing: Apply noise reduction and normalization techniques to clean the data for more precise analysis. Feature Extraction: Develop algorithms to extract the above features from the time series data. This might include detecting peaks, analyzing the rate of rise and fall in power density, and measuring durations and intervals. Classification Model: Use machine learning techniques such as decision trees, support vector machines, or neural networks to classify events based on the extracted features. Training the model with labeled examples of each event type will be crucial. Validation and Testing: Continuously validate the model with new data and adjust parameters to handle variations in signal characteristics or noise levels. By focusing on these features and employing robust data processing and machine learning techniques, you can effectively classify the types of transient electromagnetic events detected by the FORTE satellite.",
    "ECG": "To classify a patient’s heart condition based on single-lead ECG recordings effectively, various features and patterns can be extracted from the ECG signal to facilitate accurate diagnosis. Here are some key features and patterns typically considered: Heart Rate: The average heart rate can be calculated by detecting the intervals between R-peaks (RR intervals) in the ECG signal. Variations in heart rate can indicate conditions like tachycardia or bradycardia. RR Intervals: Analyzing the variability of RR intervals helps in assessing the autonomic nervous system’s control over the heart, indicating potential arrhythmias or other heart conditions. P-Wave Analysis: The presence, size, shape, and duration of the P-wave, which represents atrial depolarization, are important. Abnormalities in P-waves can indicate atrial enlargement or atrial arrhythmias. QRS Complex: The duration, amplitude, and morphology of the QRS complex, which represents ventricular depolarization, are crucial. Changes can indicate ventricular hypertrophy, bundle branch blocks, or other ventricular disorders. ST Segment: The level and shape of the ST segment can indicate ischemia or myocardial infarction. Elevation or depression of this segment is particularly significant in diagnosing these conditions. T-Wave Analysis: Alterations in T-wave morphology can be indicative of electrolyte imbalances, ischemia, or myocardial infarction. QT Interval: Measuring the duration of the QT interval, which represents the total time for ventricular depolarization and repolarization, is important. Prolonged or shortened QT intervals can lead to arrhythmias. Signal Quality: Assessing the quality of the ECG signal to detect noise, artifacts, or missing segments which could affect the analysis. Advanced Signal Processing Features: Spectral Analysis: Frequency components of the ECG can provide insights into periodic oscillations of the heart rhythm, identifying arrhythmic conditions. Wavelet Transform: This helps in detecting transient features and non-stationary changes in the ECG signal. Machine Learning Features: Feature Engineering: Creating composite features like heart rate variability, RR interval statistics (mean, median, range, standard deviation), and counts of arrhythmic beats. Time-Series Analysis: Applying algorithms to detect trends, patterns, and outliers over time. Statistical Features: These include calculating the mean, variance, skewness, and kurtosis of the intervals and amplitudes, providing a statistical summary that may indicate underlying pathologies. These features can be extracted using various signal processing techniques and then used as inputs into classification models or algorithms to determine specific heart conditions. By analyzing these aspects of the ECG, a cardiologist can effectively classify different types of heart conditions with higher accuracy.",
    "EMG": "As an Electromyograms (EMG) analysis expert tasked with determining the type of subject based on the EMG record, the following features and patterns would be essential to extract from the input data to facilitate accurate classification: Signal Amplitude: The peak amplitude of the EMG signal provides information on muscle activity level, which can vary significantly between different types of subjects, such as athletes vs. nonathletes, or among different medical conditions. Mean Absolute Value (MAV): This feature represents the average of the absolute values of the EMG signal. It is useful for estimating the overall muscle activation over time. Variance: The variance of the EMG signal can help in assessing the signal’s power and muscle fatigue, which may differentiate between subject types based on their endurance and muscle condition. Root Mean Square (RMS): This is a measure of the signal’s power, reflecting muscle force and fatigue. It’s particularly useful in continuous monitoring of muscle activity. Zero Crossing Rate (ZCR): This measures the rate at which the signal changes from positive to negative and vice versa, indicating muscle fiber recruitment patterns and firing rates. Waveform Length: The cumulative length of the waveform over time, reflecting the complexity of the muscle activation pattern. It can indicate the contractile characteristics of different muscle groups. Frequency Domain Features: Median Frequency (MDF): This frequency divides the spectrum into two regions with equal power; it shifts downwards as muscles fatigue. Mean Frequency (MNF): This is the average frequency weighted by the amplitude, used to assess  muscle fatigue and fiber composition. Power Spectral Density (PSD): Analysis of the distribution of power across various frequency bands can indicate the type of muscle activity and its intensity. Entropy: This measures the complexity or randomness of the EMG signal, useful for distinguishing between controlled and uncontrolled muscle activity. Higher Order Statistics (HOS): Skewness and kurtosis of the EMG signal provide insights into the symmetry and peakiness of the distribution, which can vary with different types of muscle activation. Autoregressive Model Coefficients: Parameters from fitting an autoregressive model to the EMG signal can help in characterizing the muscle activity and can be used as features for classification. Signal Decomposition: Wavelet Transform: Decomposing the signal into wavelets to capture both frequency and location information about muscle activity. Empirical Mode Decomposition (EMD): This non-linear and non-stationary signal analysis technique can adaptively decompose an EMG signal into intrinsic mode functions (IMFs), revealing hidden patterns. Pattern Recognition: Detecting specific patterns of muscle activation that are characteristic of certain actions or types of subjects, using sequence modeling or neural networks. These features can be extracted using advanced signal processing techniques and subsequently used in machine learning models to classify different types of subjects based on their EMG records. The choice of features and model depends on the specificity of the subjects being classified and the quality and type of EMG data available.",
    "CTU": "To differentiate between a desktop and a laptop based on 24-hour power consumption data, you would focus on extracting and analyzing specific features or patterns that can indicate the type of device based on its energy usage profile. Here are several key features and patterns you might consider: Total Daily Power Consumption: Laptops typically consume less power than desktops due to their more energy-efficient components. Calculating the total power used over a 24-hour period could give an initial indication of the device type. Power Consumption Patterns Over Time: Analyze hourly or segment-wise power consumption. Laptops might show a more uniform consumption pattern, especially if they’re left on but are in sleep or hibernate modes. Desktops might show a starker contrast between high consumption during active use and low consumption when turned off or in sleep mode. Frequency and Duration of Power Spikes: Desktops might exhibit higher power spikes during usage due to more powerful processors and peripherals compared to laptops. Observing how often and how long these spikes occur can be indicative. Minimum Power Consumption Levels: The minimum power levels (especially during inactive periods like nighttime) can be telling. Desktops might completely turn off (very low or zero consumption) or remain on higher consumption levels due to connected devices, whereas laptops typically have lower baseline consumption levels due to battery optimization features. Presence of Battery Charging Patterns: If data shows periodic drops and rises in power consumption that could correspond to a battery charging cycle, it’s likely a laptop. Desktops would not show this pattern unless a UPS or similar device is connected, which is less common. Variability in Power Consumption: Analyzing the variability and standard deviation in power usage over 24 hours can help distinguish between the two. Laptops generally have less variability in power consumption, while desktops might have greater fluctuations due to different modes of operation (idle, full power, sleep mode). Response to Day and Night Cycles: Depending on the usage patterns, if the device shows a significant reduction in power usage during typical sleeping hours, it might suggest a laptop which is often put into sleep mode automatically. Desktops might not show this pattern distinctly if they are left on for processes like downloads, updates, or backups during off-hours. By analyzing these features and considering the context in which the device is used (e.g., home, office), you can infer with reasonable accuracy whether the device is a desktop or a laptop. Statistical and machine learning models can be applied to these features to automate the classification process, especially if you have a labeled dataset to train such models.",
    "HAR": "For the task of human activity recognition based on accelerometer data along the x, y, and z axes, the extraction of relevant features is crucial for accurately classifying the type of activity. Here are some typical features and patterns you might consider extracting from the accelerometer data: Statistical Features: Mean: Average value of acceleration for each axis. Standard Deviation: Measure of the amount of variation or dispersion in the acceleration values. Variance: Squared deviation of each point from the mean, indicating the spread of the acceleration data. Median: The middle value of the data which divides the probability distribution into two equal halves. Range: Difference between the maximum and minimum values in the acceleration data. Interquartile Range (IQR): Measures the statistical dispersion as the difference between 25th and 75th percentiles. Time-domain Features: Root Mean Square (RMS): Indicates the magnitude of acceleration, computed as the square root of the average of the squares of the values. Zero Crossing Rate (ZCR): Number of times the signal changes from positive to negative and vice versa, which can indicate the frequency of the activity. Signal Magnitude Area (SMA): Integration of the magnitude of the acceleration over a window, giving a sense of the energy expenditure. Time between Peaks: Interval time between local maxima in the acceleration data. Frequency-domain Features: Fast Fourier Transform (FFT): Transforming the data from time domain to frequency domain to analyze the frequency components. Power Spectral Density (PSD): Indicates the power present in various frequency components. Spectral Entropy: Measures the regularity and complexity of the frequency distribution. Peak Frequency: The frequency with the maximum power, indicating the dominant frequency of movement. Correlationbased Features: Correlation between Axes: Measures how related the movements in different axes are, which can indicate coordinated motion patterns. Geometrical Features: Angles between Axes: Can help in understanding the orientation of the body in space. Magnitude of Acceleration Vector: Calculated as the square root of the sum of the squares of x, y, and z components. It provides a holistic view of the acceleration independent of the direction. Entropy-based Features: Signal Entropy: Indicates the unpredictability or complexity of the acceleration signal. By extracting these features from the accelerometer data, you can capture a comprehensive profile of the movements, which can then be fed into a machine learning model to classify different types of human activities such as walking, running, sitting, standing, etc.",
    "HAD": """To successfully classify human activities based on 6-axis IMU data (tri-axial accelerometer and tri-axial gyroscope) from a wearable MotionNode sensor, you would analyze specific features from the time series that correspond to the unique characteristics of each activity type. Here are the key features and patterns to focus on for each of the twelve activity types:

                **Walking Forward (Activity 1):**
                - Periodic Pattern: Look for regular, rhythmic oscillations in all axes, particularly noticeable in the vertical (Z) accelerometer axis during heel strikes.
                - Frequency: Typical walking frequency is around 1-2 Hz (60-120 steps per minute).
                - Gyroscope Pattern: Moderate angular velocity changes primarily in the sagittal plane (pitch rotation).
                - Acceleration Range: Moderate magnitude acceleration (typically 1-2g peak-to-peak in vertical axis).

                **Walking Left (Activity 2) and Walking Right (Activity 3):**
                - Lateral Movement: Increased acceleration in the Y-axis (side-to-side) compared to forward walking.
                - Body Rotation: Distinct gyroscope patterns showing rotation around the vertical axis (yaw).
                - Asymmetric Pattern: Different from forward walking with more pronounced lateral sway.

                **Walking Upstairs (Activity 4):**
                - Increased Vertical Acceleration: Higher Z-axis acceleration peaks during step-up motion.
                - Higher Energy: Greater overall acceleration magnitude compared to level walking.
                - Gyroscope: Forward tilt (pitch) patterns as body leans forward.
                - Frequency: Slower cadence than level walking (typically 0.8-1.5 Hz).

                **Walking Downstairs (Activity 5):**
                - Sharp Deceleration: Distinct downward acceleration spikes followed by impact absorption.
                - Negative Z-axis Peaks: More pronounced negative acceleration during descent.
                - Gyroscope: Backward tilt compensation patterns.
                - Higher Impact Forces: Greater instantaneous acceleration magnitudes at foot strikes.

                **Running Forward (Activity 6):**
                - Higher Frequency: Faster cadence than walking (typically 2-3 Hz).
                - Larger Amplitude: Much higher acceleration magnitudes (can exceed 3-4g).
                - Flight Phase: Brief periods of near-zero acceleration between foot strikes.
                - High Gyroscope Activity: Rapid angular velocity changes due to faster movement.

                **Jumping Up (Activity 7):**
                - Explosive Pattern: Sharp, high-magnitude vertical acceleration spike during takeoff.
                - Zero-G Phase: Brief weightlessness period during flight (acceleration near 0g).
                - Impact Landing: Very high acceleration spike upon landing (can exceed 5-6g).
                - Gyroscope Burst: Sudden angular velocity changes during jump preparation and landing stabilization.

                **Sitting (Activity 8):**
                - Minimal Movement: Very low acceleration variance across all axes (close to gravity baseline).
                - Static Gyroscope: Near-zero angular velocity in all axes.
                - Gravity Vector: Stable 1g reading in the vertical axis aligned with body orientation.
                - Occasional Micro-movements: Small adjustments or fidgeting but overall low energy.

                **Standing (Activity 9):**
                - Low Variance: Slightly higher variance than sitting due to postural sway.
                - Gravity Dominant: Steady 1g acceleration in vertical axis.
                - Postural Sway: Small, slow oscillations in accelerometer (< 0.2 Hz).
                - Minimal Gyroscope: Very low angular velocity with occasional small corrections.

                **Sleeping (Activity 10):**
                - Extremely Low Activity: Minimal acceleration changes, mostly gravity vector.
                - Rare Movements: Occasional large movements (position changes) separated by long static periods.
                - Low Frequency: Any movements are very slow and infrequent.
                - Stable Orientation: Long periods of constant sensor orientation relative to gravity.

                **Elevator Up (Activity 11):**
                - Smooth Vertical Acceleration: Gradual positive Z-axis acceleration at start, then steady.
                - Constant Velocity Phase: Period of steady 1g acceleration with minimal variation.
                - Deceleration: Smooth negative acceleration before stopping.
                - Minimal Gyroscope: Very low angular velocity (passive vertical transport).

                **Elevator Down (Activity 12):**
                - Initial Deceleration: Slight negative Z-axis acceleration at start.
                - Constant Descent: Steady acceleration pattern during constant velocity descent.
                - Braking Pattern: Positive Z-axis acceleration spike before stopping.
                - Similar to Elevator Up: But with inverted acceleration profile.

                **Feature Extraction Strategies:**

                **Time-Domain Features:**
                - Statistical Measures: Mean, standard deviation, variance, range, min/max for each axis.
                - Signal Magnitude Area (SMA): Overall movement intensity.
                - Energy: Sum of squared values indicating activity level.
                - Zero-Crossing Rate: Frequency of signal oscillations.
                - Correlation: Between accelerometer axes and between accelerometer/gyroscope.

                **Frequency-Domain Features:**
                - FFT Coefficients: Dominant frequencies for periodic activities.
                - Spectral Entropy: Signal regularity/complexity.
                - Power Spectral Density: Energy distribution across frequencies.
                - Peak Frequency: Dominant motion frequency (cadence for walking/running).

                **Time-Frequency Features:**
                - Wavelet Transform: Capture transient patterns like jumps or impacts.
                - Short-Time Fourier Transform: Analyze frequency changes over time.

                **Sensor Fusion Features:**
                - Magnitude Vector: √(x² + y² + z²) for both accelerometer and gyroscope.
                - Euler Angles: Body orientation estimated from sensor fusion.
                - Jerk: Rate of change of acceleration (sudden movements).

                **Classification Approach:**
                1. **Preprocessing**: Apply noise filtering (low-pass filter at ~20Hz) and normalize data.
                2. **Segmentation**: Use sliding windows (e.g., 128 samples = 1.28s at 100Hz with 50% overlap).
                3. **Feature Engineering**: Extract comprehensive feature set from each window.
                4. **Model Training**: Use machine learning classifiers (Random Forest, SVM, CNN, LSTM) with labeled training data.
                5. **Validation**: Use subject-wise cross-validation to ensure generalization to new users.

                **Key Discriminative Patterns:**
                - **Static vs Dynamic**: Use total acceleration variance (sitting/standing/sleeping vs moving activities).
                - **Periodic vs Aperiodic**: FFT analysis separates walking/running from jumping/elevator.
                - **Vertical Transport**: Gyroscope inactivity + smooth Z-acceleration patterns identify elevator.
                - **High Impact**: Peak acceleration magnitude distinguishes jumping and running from walking.
                - **Directional Movement**: Axis-specific patterns differentiate walking directions.
                - **Frequency Content**: Cadence separates walking from running and identifies activity tempo.

                By systematically analyzing these temporal and spectral features across both accelerometer and gyroscope channels, you can effectively classify the twelve USC-HAD activities with high accuracy."""
}

NO_CoT_QUESTION_TAG = (
    'Answer using exactly this format: "The answer is [X] [CLASS_NAME]". '
    'Example: "The answer is [D] CABBAGE".\n'
)

CoT_QUESTION_TAG = (
    "Think step by step and explain your reasoning. "
    "Afterward, provide the final answer using exactly this format: "
    '"The answer is [X] [CLASS_NAME]". '
    'Example: "The answer is [D] CABBAGE".'
)

TITLE_MAPPINGS["trHARteHAD".upper()] = TITLE_MAPPINGS["HAR"]  # same dataset, just different split
TASK_DESCRIPTION["trHARteHAD".upper()] = TASK_DESCRIPTION["HAR"]  # same dataset, just different split
LABEL_MAPPING["trHARteHAD".upper()] = LABEL_MAPPING["HAR"]
X_MAPPINGS["trHARteHAD".upper()] = X_MAPPINGS["HAR"]
Y_MAPPINGS["trHARteHAD".upper()] = Y_MAPPINGS["HAR"]
LEGEND_MAPPINGS["trHARteHAD".upper()] = LEGEND_MAPPINGS["HAR"]
EXTRA_INFO_MAPPINGS["trHARteHAD".upper()] = EXTRA_INFO_MAPPINGS["HAR"]
HELP_STRING["trHARteHAD".upper()] = HELP_STRING["HAR"]