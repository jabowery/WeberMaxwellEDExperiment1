###
##
# Simulate processing of the 2 signals as received by the Picoscope in:
# Experimental detection of superluminal far-field radio waves with transverse plasma antennas
# by Steffen Kuhn
#
# The simulated signals include the following components:
#
# Frequencies phase shifted by 100ns due to lamps' high velocity electrons:
# 1. Audible frequencies from FM station.
# 2. FM stereo pilot frequency at 19kHz
#
# Frequencies NOT phase shifted due to being picked up by wires' low velocity electrons:
# 1. Exponential distribution of frequencies higher than pilot frequency.
# 2. 60Hz buzz from power sources.
#
# Processing of superposition of those frequency components into 2 signals:
# 1. 5MHz sampling frequency quantization
# 2. Centered about the mean (simulation of DC subtract).
# 3. Limited to 4 standard deviations (hits the rail and is clipped).
# 4. 8-bit signed integer quantization.
# 5. 10kHz low pass.
# 6. Downsampled to 50kHz.
# 7. Upsample to 100MHz using (fft) Whittaker-Shannon interpolation
# 8. Find phase shift with maximum correlation.
#
# This should output 100ns phase shift.
# And it does.
##
###

import argparse

# Create the parser
parser = argparse.ArgumentParser(description="PicoScope signal processing parameters")

# Add arguments
parser.add_argument('--upsampleFrequency', type=int, default=int(1e8), 
                    help='Sample frequency used in cross-correlation to detect phase difference (samples/sec). (default %(default)d)')
parser.add_argument('--numComponents', type=int, default=512, 
                    help='How many randomly chosen frequencies the PicoScope sees for each of 2 electron velocity types: plasmas and wires. (default %(default)d)')
parser.add_argument('--duration', type=float, default=1.0, 
                    help='How many seconds the PicoScope samples incoming signals. (default %(default)d)')
parser.add_argument('--downsampleRate', type=int, default=50000, 
                    help='Sample frequency used to create .wav file for further analysis (samples/sec). (default %(default)d)')
parser.add_argument('--sampleRate', type=int, default=5000000, 
                    help='Sample frequency of the PicoScope. (default %(default)d)')
parser.add_argument('--nanosecondTimeShift', type=float, default=100, 
                    help='Total difference in phase between forward and backward lamp electron velocities (nanoseconds) (default %(default)d)')
parser.add_argument('--highestAudibleFrequency', type=int, default=15000, 
                    help='High frequency limit on the output of the FM radios other than pilot tone (Hz). (default %(default)d)')
parser.add_argument('--cutoffFrequency', type=int, default=10000, 
                    help='Frequency for low pass audio filter. (default %(default)d)')
parser.add_argument('--ampCutoffSTD', type=int, default=4, 
                    help='Amplitude cutoff for signal scaling to 8-bit signed samples (standard deviations). (default %(default)d)')
parser.add_argument('--maxProcessors', type=int, default=None, 
                    help='Maximum number of processors (default is 1/2 the number of processors on your system, else 1)')

# Parse the arguments
args = parser.parse_args()

# Assign variables
upsampleFrequency = args.upsampleFrequency
numComponents = args.numComponents
duration = args.duration
#duration = 1
downsampleRate = args.downsampleRate
sampleRate = args.sampleRate
nanosecondTimeShift = args.nanosecondTimeShift
highestAudibleFrequency = args.highestAudibleFrequency
cutoffFrequency = args.cutoffFrequency
ampCutoffSTD = args.ampCutoffSTD
maxProcessors = args.maxProcessors

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
import sys
from multiprocessing import Pool, cpu_count

def sinusoid_component(args):
    freq, phase, t = args
    return np.sin(2 * np.pi * freq * (t + phase))

def generate_signal_parallel(t, frequencies, phases):
    if maxProcessors:
        num_processes = maxProcessors
    else:
        num_processes = int(cpu_count()/2)
        num_processes = 1 if num_processes==0 else num_processes
#    with Pool(num_processes) as pool:
    from concurrent import futures
    with futures.ProcessPoolExecutor(max_workers=num_processes) as pool:
        args = [(freq, phase, t) for freq, phase in zip(frequencies, phases)]
        results = pool.map(sinusoid_component, args)
        
    result = np.zeros(len(t))
    for res in results:
        result += res
    
    return result

def save_signal(signal, filename):
    np.save(filename, signal)

def load_signal(filename):
    return np.load(filename + '.npy')

#print("Signal data:")
#print(signal)


# Example usage
#frequencies = np.random.rand(64)  # Random frequencies
#phases = np.random.rand(64)       # Random phases
#t = np.linspace(0, 1, 1000)  # Reduced number of time points for demonstration

# Generate the signal
#signal = generate_signal(frequencies, phases, t)


def sinc_interpolation_fft(x: np.ndarray, s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Fast Fourier Transform (FFT) pilotd sinc or bandlimited interpolation.

    Args:
        x (np.ndarray): signal to be interpolated, can be 1D or 2D
        s (np.ndarray): time points of x (*s* for *samples*)
        u (np.ndarray): time points of y (*u* for *upsampled*)

    Returns:
        np.ndarray: interpolated signal at time points *u*
    """
    num_output = len(u)

    # Compute the FFT of the input signal
    X = np.fft.rfft(x)

    # Create a new array for the zero-padded frequency spectrum
    X_padded = np.zeros(num_output // 2 + 1, dtype=complex)

    # Copy the original frequency spectrum into the zero-padded array
    X_padded[:X.shape[0]] = X

    # Compute the inverse FFT of the zero-padded frequency spectrum
    x_interpolated = np.fft.irfft(X_padded, n=num_output)

    return x_interpolated * (num_output / len(s))

def plotXYandXZ(XValues, YValues, ZValues, label="",start=None,end=None):
    lX=len(XValues)
    
    # Optional: Plot the interpolated signals
    plt.figure(figsize=(10, 6))
    plt.plot(XValues[start:end], YValues[:lX][start:end], label=label+' Signal 1')
    plt.plot(XValues[start:end], ZValues[:lX][start:end], label=label+' Signal 2', alpha=0.7)
    plt.legend()
    plt.show()



# Force first frequency to be pilotFrequency
#maxFrequencyIndex = np.argmin(np.abs(frequencies - pilotFrequency))

###
##
# Define signals seen by scope as output of FM radio.
###
##
# Define phase-shifted audible signals from FM station.
frequencies = np.random.random(numComponents)*highestAudibleFrequency
#frequencies[abs(frequencies-pilotFrequency)<pilotFrequency*.1] = 15000 #exclude frequencies within 10% of pilotFrequency by making them highest audible in FM radio
rphases = np.random.random(len(frequencies))
timeShift = nanosecondTimeShift/1e9
phases1 = rphases - timeShift/2
phases2 = rphases + timeShift/2
# Phase-shifted audible signals from FM station defined.
##
###
##
# Define the pilot frequency 19kHz
pilotFrequency = 19000  # Frequency in Hz
IndexOfPilotFrequency = 0
frequencies[IndexOfPilotFrequency] = pilotFrequency
# The Pilot frequency 19kHz defined.
##
###
##
# Define noise from wires between FM radio and scope.
# (No phase difference due to low wire electron velocities.)
phases1 = np.append(phases1,rphases)
phases2 = np.append(phases2,rphases)
###
##
# Define high frequency noise in the wires of the measurement apparatus
noiseFrequencies = np.exp(13+np.random.random(len(frequencies))*5)
frequencies = np.append(frequencies,noiseFrequencies)
# High frequency noise in the wires of the measurement apparatus defined
##
###
##
# Define 60Hz buzz.
IndexOf60HzBuzz = len(frequencies)-len(noiseFrequencies)
frequencies[IndexOf60HzBuzz] = 60 # 60hz buzz
phases1[IndexOf60HzBuzz] = np.random.random()
phases2[IndexOf60HzBuzz] = phases1[IndexOf60HzBuzz]
# 60Hz buzz defined.
##
###
# Noise from wires between FM radio and scope defined.
##
###
# Signals seen by scope as output of FM radio defined.
##
###
print(frequencies)
print(phases1)
print(phases2)

# Function to generate signals
#def generate_signal(t, frequencies, phases):
#    return np.sum(np.sin(2 * np.pi * frequencies[:, None] * (t + phases[:, None])), axis=0)

# Sampling rate and duration

timePoints = np.linspace(0, duration, np.rint(sampleRate*duration).astype(int), False)
#timePoints = np.linspace(0, duration, sampleRate*duration, False)

filename1 = f'signal{len(frequencies)}1'
filename2 = f'signal{len(frequencies)}2'

if os.path.exists(filename1 + '.npy'):
    signal1 = load_signal(filename1)
    signal2 = load_signal(filename2)
else:
    # Generate the signal using parallel processing
    signal1 = generate_signal_parallel(timePoints,frequencies, phases1)
    # Save the signal to a file
    save_signal(signal1, filename1)
    signal2 = generate_signal_parallel(timePoints,frequencies, phases2)
    # Save the signal to a file
    save_signal(signal2, filename2)

def sample8bitsigned(signal, stds=ampCutoffSTD):
    # Return an 8 bit signed version of signal after performing these processing steps:
    # Center the signal about the mean.
    # Any values that lie outside of stds standard deviations of the mean, set to the closest value within that limit.
    # Scale to the range -127 to 127.
    # Round to the nearest integer.
    # return the resulting signal.A


    # Calculate the mean and standard deviation of the signal
    mean = np.mean(signal)
    std_dev = np.std(signal)
    
    # Center the signal about the mean
    centered_signal = signal - mean
    
    # Clip values to within stds standard deviations of the mean
    lower_bound = -stds * std_dev
    upper_bound = stds * std_dev
    clipped_signal = np.clip(centered_signal, lower_bound, upper_bound)
    
    # Scale to the range -127 to 127
    # First find the max absolute value in the clipped signal to scale appropriately
    max_val = max(abs(lower_bound), abs(upper_bound))
    scaled_signal = (clipped_signal / max_val) * 127
    
    # Round to the nearest integer
    rounded_signal = np.rint(scaled_signal).astype(int)
    
    # Ensure values are within the 8-bit signed integer range
    final_signal = np.clip(rounded_signal, -127, 127)
    
    return final_signal

# Example usage
#signal = np.random.normal(0, 1, 1000)  # Generate a random signal with mean 0 and std deviation 1
#sampled_signal = sample8bitsigned(signal, 3)  # Process the signal with 3 standard deviations

#print("Sampled signal:")
#print(sampled_signal)


# Sample both signals
sampledSignal1 = sample8bitsigned(signal1)
signal1 = None
sampledSignal2 = sample8bitsigned(signal2)
signal2 = None
#plotXYandXZ(timePoints, sampledSignal1, sampledSignal2, "Raw",0,int(8*sampleRate/pilotFrequency))


def plotlp(): #Hz
# Apply a low-pass filter to reduce frequencies above 10 kHz
    sos = signal.butter(4, cutoffFrequency*2, btype='low', fs=sampleRate, output='sos') # butter takes nyquist frequency
    filteredSignal1 = signal.sosfilt(sos, sampledSignal1)
    filteredSignal2 = signal.sosfilt(sos, sampledSignal2)
    plotXYandXZ(timePoints, filteredSignal1, filteredSignal2, "Filtered")
#breakpoint()
# Apply a low-pass filter to reduce frequencies above 10 kHz
sos = signal.butter(4, cutoffFrequency*2, btype='low', fs=sampleRate, output='sos') # butter takes nyquist frequency
filteredSignal1 = signal.sosfilt(sos, sampledSignal1)
sampledSignal1 = None
filteredSignal2 = signal.sosfilt(sos, sampledSignal2)
sampledSignal2 = None


#plotXYandXZ(timePoints[:t2plot], filteredSignal1, filteredSignal2, "Filtered")

# Downsample to 50 kHz
downsampleFactor = int(sampleRate / downsampleRate)
downsampledSignal1 = filteredSignal1[::downsampleFactor]
filteredSignal1 = None
downsampledSignal2 = filteredSignal2[::downsampleFactor]
filteredSignal2 = None
downsampledTimePoints = timePoints[::downsampleFactor]
#plotXYandXZ(downsampledTimePoints, downsampledSignal1, downsampledSignal2, "Downsampled")

signal1 = downsampledSignal1
signal2 = downsampledSignal2

# Define sinc function to avoid division by zero
def sinc(t, ts):
    return np.where(t == ts, 1, np.sin(np.pi * (t - ts)) / (np.pi * (t - ts)))

# Perform Whittaker-Shannon interpolation
def pilotSamples(sampf):
    return int(sampf/pilotFrequency)

# Sample points for plotting
#    Args:
#        x (np.ndarray): signal to be interpolated, can be 1D or 2D
#        s (np.ndarray): time points of x (*s* for *samples*)
#        u (np.ndarray): time points of y (*u* for *upsampled*)
upsampledTimePoints = np.linspace(0, duration, np.rint(upsampleFrequency*duration).astype(int), False)
interpolatedSignal1 = sinc_interpolation_fft(signal1, downsampledTimePoints, upsampledTimePoints)
bp1 = None
interpolatedSignal2 = sinc_interpolation_fft(signal2, downsampledTimePoints, upsampledTimePoints)
bp2 = None
halfwaySample = int(len(upsampledTimePoints)/2)
# Plot 3 cycles of the pilot frequency.
plotXYandXZ(upsampledTimePoints, interpolatedSignal1, interpolatedSignal2,"Interpolated",halfwaySample,halfwaySample+pilotSamples(upsampleFrequency)*3)

corr12 = signal.correlate(interpolatedSignal1, interpolatedSignal2, mode='full')
lags = np.arange(-len(interpolatedSignal2) + 1, len(interpolatedSignal2))
lag_max = lags[np.argmax(corr12)]*(1e9/upsampleFrequency)
print(lag_max,'ns phase shift')
print(lag_max,'ns phase shift',file=sys.stderr)
