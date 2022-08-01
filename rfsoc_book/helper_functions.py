import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal

def plot_iq_timeseries(x, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.plot(np.real(x), **kwargs)
    ax.plot(np.imag(x), **kwargs)
    ax.set_title('Time series plot')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Magnitude')
    ax.legend(('Real', 'Imaginary'))
    return ax

def frequency_plot(freqs, signal_db, ax=None):
    if ax is None:
        ax = plt.gca()
    plt.plot(freqs, signal_db)
    plt.title('FFT plot')
    plt.xlabel('Frequency, Hz')
    plt.ylabel('Amplitude, dB')
    return ax

def scatterplot(x, y, ax=None):
    if ax is None:
        plt.figure(figsize=(5,5))
        ax = plt.gca()
    ax.scatter(x,y)
    ax.set_title('Constellation plot')
    ax.set_xlabel('Channel 1 amplitude')
    ax.set_ylabel('Channel 2 amplitude')
    return ax

def scatterplot_with_ref(x, x_ref, ax=None):
    if ax is None:
        plt.figure(figsize=(5,5))
        ax = plt.gca()
    ax.scatter(np.real(x), np.imag(x))
    ax.scatter(np.real(x_ref), np.imag(x_ref), s=120, c='red', marker='x')
    ax.set_title('Constellation plot')
    ax.set_xlabel('Channel 1 amplitude')
    ax.set_ylabel('Channel 2 amplitude')
    ax.legend(('Received symbols', 'Reference points'), \
           bbox_to_anchor=(1, 1), loc='upper left')
    return ax

def subplots(signal, fft_signal, samples, freqs, fs, title1, title2):
    #plot time and freq domain subplots
    fig, axes = plt.subplots(1,2, figsize = (12,4))
    x = np.arange(1,11)
    #axes[0].plot(samples[0:500],signal[0:500])
    axes[0].plot(samples[0:100],signal[0:100])
    axes[0].grid(True)
    axes[0].set_title(title1)
    axes[0].set_xlabel('Samples')
    axes[0].set_ylabel('Amplitude')
    axes[1].plot(freqs[int(fs/2):]/1000, fft_signal[int(fs/2):])
    axes[1].grid(True)
    axes[1].set_title(title2)
    axes[1].set_xlabel('Frequency, KHz')
    axes[1].set_ylabel('Amplitude, dB')
    fig.tight_layout()
    plt.show()
    
    return

def multi_subplots(signal_I, signal_Q, fft_signal_I, fft_signal_Q, samples, freqs, fs, title1, title2, label_I, label_Q, legend_title):
    #plot I and Q in time and freq domains
    fig, axes = plt.subplots(1,2, figsize = (12,4))
    x = np.arange(1,11)
    axes[0].plot(samples[0:100],signal_I[0:100], label = label_I)
    axes[0].plot(samples[0:100],signal_Q[0:100], label = label_Q)
    axes[0].grid(True)
    axes[0].set_title(title1)
    axes[0].set_xlabel('Samples')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(title = legend_title)
    
    axes[1].plot(freqs[int(fs/2):]/1000, fft_signal_I[int(fs/2):], label=label_I)
    axes[1].plot(freqs[int(fs/2):]/1000, fft_signal_Q[int(fs/2):], label=label_Q)
    axes[1].grid(True)
    axes[1].set_title(title2)
    axes[1].set_xlabel('Frequency, KHz')
    axes[1].set_ylabel('Amplitude, dB')
    axes[1].legend(title = legend_title)
    
    fig.tight_layout()
    plt.show()
    
    return

def complex_subplots(signal_I, signal_Q, fft_signal, samples, freqs, fs, label_I, label_Q, title1, title2, legend_title):
    #plot I and Q in time, single plot in freq
    fig, axes = plt.subplots(1,2, figsize = (12,4))
    x = np.arange(1,11)
    axes[0].plot(samples[0:100],signal_I[0:100], label = label_I)
    axes[0].plot(samples[0:100],signal_Q[0:100], label = label_Q)
    axes[0].grid(True)
    axes[0].set_title(title1)
    axes[0].set_xlabel('Samples')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(title = legend_title)
    
    axes[1].plot(freqs[int(fs/2):]/1000, fft_signal[int(fs/2):])
    axes[1].grid(True)
    axes[1].set_title(title2)
    axes[1].set_xlabel('Frequency, KHz')
    axes[1].set_ylabel('Amplitude, dB')
    fig.tight_layout()
    plt.show()
    
    return

def find_fft(signal, n_window):
    fft = np.fft.fftshift(np.fft.fft(signal, n_window))

    return 10*np.log10(abs(fft)/len(signal))

def tripleSubplots(fft1, fft2, fft3, samples, freqs, fs, title1, title2, title3):
    fig, axes = plt.subplots(1,3, figsize = (12,4))
    size = np.arange(1,11)
    axes[0].plot(freqs[int(fs/2):], fft1[int(fs/2):])
    axes[0].grid(True)
    axes[0].set_title(title1)
    axes[0].set_xlabel('Frequency, Hz')
    axes[0].set_ylabel('Amplitude, dB')
    axes[1].plot(freqs[int(fs/2):], fft2[int(fs/2):])
    axes[1].grid(True)
    axes[1].set_title(title2)
    axes[1].set_xlabel('Frequency, Hz')
    axes[1].set_ylabel('Amplitude, dB')
    axes[2].plot(freqs[int(fs/2):], fft3[int(fs/2):])
    axes[2].grid(True)
    axes[2].set_title(title3)
    axes[2].set_xlabel('Frequency, Hz')
    axes[2].set_ylabel('Amplitude, dB')
    fig.tight_layout()
    plt.show()
    
    return

# Function to calculate Power Spectral Density (PSD)
# Uses Welch's average periodogram method

# Input: signal of interest, fft size, window type 
# and overlap ratio. 

# Output: PSD estimate linear scale 

def psd(sig,nfft,wtype,overlap):

    # Input checking 
    if (wtype != 'Hamming') and (wtype != 'Bartlett') \
        and (wtype != 'Blackman') and (wtype != 'Hann')\
        and (wtype != 'Rectangle'):  
        
        raise Exception('Window must be: Hamming, Bartlett, Blackman, Hann or Rectangle.')   
        
    if (overlap < 0) or (overlap > 1):
        
        raise Exception('Overlap ratio must be between 0 and 1.')
        
    # Compute window coefficients 
    if wtype == 'Hamming':
        window = signal.windows.hamming(nfft,0)
    elif wtype == 'Bartlett':
        window = signal.windows.bartlett(nfft,0)
    elif wtype ==  'Blackman':
        window = signal.windows.blackman(nfft,0)
    elif wtype ==  'Hann':
        window = signal.windows.hann(nfft,0)
    elif wtype == 'Rectangle':
        window = signal.windows.boxcar(nfft)
        
    # Calculate no. of overlap samples
    noverlap = math.floor(nfft*overlap)
    
    # No. of FFTs to evaluate 
    length = sig.size
    num_fft = (length - noverlap) // (nfft - noverlap)
    
    # Initialise vector to hold periodograms 
    pgram_vec = np.zeros(num_fft*nfft)
    j = 0 
    k = 0 
    
    for i in range(num_fft):
        
        # Compute periodogram using sliding window  
        pgram_vec[j:j+nfft] = np.abs(np.fft.fft(sig[k:k+nfft],nfft) * window) ** 2
        
        # Increment j & k 
        j = j + nfft
        k = k + (nfft - noverlap)
       
    # Reshape periodogram vector 
    pgram_res = pgram_vec.reshape(num_fft,nfft)
    
    # Average over 1st dimension to get PSD estimate   
    psd_est = np.sum(pgram_res,0,np.float64)/num_fft
    
    return psd_est
	
def awgn(signal,SNR):
    # Measure signal power 
    s_p = np.mean(abs(signal)**2)
    
    # Calculate noise power 
    n_p = s_p/(10 **(SNR/10))
    
    # Generate complex noise 
    noise = np.sqrt(n_p/2)*(np.random.randn(len(signal)) + \
                                    np.random.randn(len(signal))*1j)
    
    # Add signal and noise 
    signal_noisy = signal + noise 
    
    return signal_noisy    
	
# Function to generate a block of BPSK, QPSK or 16-QAM symbols
def symbol_gen(nsym,mod_scheme):
    
    if mod_scheme == 'BPSK':
        # 1 bit per symbol for BPSK 
        m = 1  
        M = 2 ** m 
    
        # BPSK symbol values
        bpsk = [-1+0j, 1+0j]
        
        # Generate random integers 
        ints = np.random.randint(0,M,nsym)
        
        # Generate BPSK symbols 
        data = [bpsk[i] for i in ints]
        data = np.array(data,np.complex64)

    elif mod_scheme == 'QPSK': 
        # 2 bits per symbol for QPSK 
        m = 2
        M = 2 ** m 
    
        # QPSK symbol values 
        qpsk = [1+1j, -1+1j, 1-1j, -1-1j] / np.sqrt(2)
        
        # Generate random integers 
        ints = np.random.randint(0,M,nsym)
    
        # Map to QPSK symbols 
        data = [qpsk[i] for i in ints]
        data = np.array(data,np.complex64)
        
    elif mod_scheme == '16-QAM': 
        # 4 bits per symbol for 16-QAM 
        m = 4 
        M = 2 ** m 
        
        # 16-QAM symbol values  
        qam16 = [-3-3j, -3-1j, -3+3j, -3+1j,  \
                -1-3j, -1-1j, -1+3j, -1+1j,  \
                 3-3j,  3-1j,  3+3j,  3+1j,  \
                 1-3j,  1-1j,  1+3j,  1+1j] / np.sqrt(10)
        
        # Generate random integers 
        ints = np.random.randint(0,M,nsym)
        
        # Map to 16-QAM symbols 
        data = [qam16[i] for i in ints]
        data = np.array(data,np.complex64)
        
    else: 
        raise Exception('Modulation method must be BPSK, QPSK or 16-QAM.')
    
    return data 

def calculate_evm(symbols_tx, symbols_rx):
    evm_rms = np.sqrt(np.mean(np.abs(symbols_rx - symbols_tx )**2)) / \
              np.sqrt(np.mean(np.abs(symbols_tx)**2))
    
    return evm_rms*100


def plot_timeseries(title, x, y, line=['continuous']):
    """Function to plot up multiple timeseries on the same axis.
    
    Parameters
    ----------
    title : string
        The title of the plot.
    x : numerical array
        An array containing the x axes of the plots.
    y : numerical array
        An array containing the y axes of the plots.
    line : string array
        An array stating the plot type for each plot.
        (Continuous, discrete, dash)
    """
    plt.figure(figsize=(10,4))
    ax = plt.gca()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title(title)

    for i in range(len(x)):
        if line[i] == 'continuous':
            ax.plot(x[i],y[i])
        elif line[i] == 'discrete':
            ax.stem(x[i], y[i], basefmt='blue', linefmt='red', markerfmt='C3o', use_line_collection=True)
        elif line[i] == 'dash':
            ax.plot(x[i],y[i],'green',linestyle='--',linewidth = 2.5)
    
    plt.show()

def plot_response(fs, w, h, title, xlim=None):
    """Utility function to plot response functions
    """
    if xlim == None:
        xlim = fs/2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(0.5*fs*w/np.pi, 20*np.log10(abs(h)))
    ax.set_ylim(-60, 20)
    ax.set_xlim(0, xlim)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (dB)')
    ax.set_title(title)
    
def plot_fft(freqs, fft_signal, fs, title, label=None):
    plt.figure(figsize=(10,4))
    ax = plt.gca()
    for i in range(len(freqs)):
        if label != None:
            ax.plot(freqs[i][int(len(freqs[i])/2):], 
                    fft_signal[i][int(len(freqs[i])/2):],
                    label=label[i])
        else:
            ax.plot(freqs[i][int(len(freqs[i])/2):], 
                    fft_signal[i][int(len(freqs[i])/2):])
        
    plt.axvline(x = fs, color = 'g', linestyle='--', label='fs')
    plt.axvline(x = fs/2, color = 'r', linestyle='--', label='fs/2')
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel('Frequency, Hz')
    ax.set_ylabel('Amplitude, dB')
    ax.set_xlim(0, 3000)
    ax.legend()
    plt.show()
    
def plot_fft(freqs, fft_signal, fs, title, label=None):
    plt.figure(figsize=(10,4))
    ax = plt.gca()
    for i in range(len(freqs)):
        if label != None:
            ax.plot(freqs[i][int(len(freqs[i])/2):], 
                    fft_signal[i][int(len(freqs[i])/2):],
                    label=label[i])
        else:
            ax.plot(freqs[i][int(len(freqs[i])/2):], 
                    fft_signal[i][int(len(freqs[i])/2):])
        
    plt.axvline(x = fs, color = 'g', linestyle='--', label='fs')
    plt.axvline(x = fs/2, color = 'r', linestyle='--', label='fs/2')
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel('Frequency, Hz')
    ax.set_ylabel('Amplitude, dB')
    ax.set_xlim(0, 3000)
    ax.legend()
    plt.show()
    
    
### Machine learning helpers ###

# Function to generate BPSK
def generate_bpsk(num_symbols, noise=50):
    bits = np.random.randint(0,2,num_symbols)
    bpsk_scheme = [1+0j, -1+0j]
    bpsk_symbols = np.array([bpsk_scheme[i] for i in bits])
    
    bpsk_symbols = awgn(bpsk_symbols, noise)
    
    return bpsk_symbols

# Function to generate QPSK
def generate_qpsk(num_symbols, noise=50):
    qpsk_scheme= [1+1j, 1-1j, -1+1j, -1-1j]
    ints = np.random.randint(0,4,num_symbols)
    qpsk_symbols = np.array([qpsk_scheme[i] for i in ints])/np.sqrt(2)

    qpsk_symbols = awgn(qpsk_symbols, noise)
    
    return qpsk_symbols

# Function to generate QAM
def generate_qam(num_symbols, noise=50):
    qam_scheme = [-3-3j, -3-1j, -3+3j, -3+1j,  \
                  -1-3j, -1-1j, -1+3j, -1+1j,  \
                   3-3j,  3-1j,  3+3j,  3+1j,  \
                   1-3j,  1-1j,  1+3j,  1+1j]
    ints = np.random.randint(0,16,num_symbols)
    qam_symbols = np.array([qam_scheme[i] for i in ints])
    qam_symbols = qam_symbols/np.mean(np.abs(qam_scheme))
    
    qam_symbols = awgn(qam_symbols, noise)
    
    return qam_symbols

# Function to generate 4-ASK
def generate_ask4(num_symbols, noise=50):
    ask4_scheme = [3+0j, 1+0j, -1+0j, -3+0j]
    ints = np.random.randint(0,4,num_symbols)
    ask4_symbols = np.array([ask4_scheme[i] for i in ints])
    ask4_symbols = ask4_symbols/np.mean(np.abs(ask4_scheme))
    
    ask4_symbols = awgn(ask4_symbols, noise)
    
    return ask4_symbols

def calculate_statistics(x):
    
    # Extract instantaneous amplitude and phase
    inst_a = np.abs(x) 
    inst_p = np.angle(x)

    # Amplitude statistics
    m2_a = np.mean((inst_a-np.mean(inst_a))**2) # variance of amplitude
    m3_a = np.mean((inst_a-np.mean(inst_a))**3)/(np.mean((inst_a-np.mean(inst_a))**2)**(3/2)) # skewness of amplitude
    m4_a = np.mean((inst_a-np.mean(inst_a))**4)/(np.mean((inst_a-np.mean(inst_a))**2)**(2)) # kurtosis of amplitude
    
    # Phase statistics
    m2_p = np.mean((inst_p-np.mean(inst_p))**2) # variance of phase
    m3_p = np.mean((inst_p-np.mean(inst_p))**3)/(np.mean((inst_p-np.mean(inst_p))**2)**(3/2)) # skewness of phase
    m4_p = np.mean((inst_p-np.mean(inst_p))**4)/(np.mean((inst_p-np.mean(inst_p))**2)**(2)) # kurtosis of phase
    
    return  m2_a, m3_a, m4_a, m2_p, m3_p, m4_p