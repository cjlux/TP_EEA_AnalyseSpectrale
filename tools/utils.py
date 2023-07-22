#
# v1.1 Juin  2023, by JLC
#

import numpy as np
from numpy.fft import rfft    # rfft : FFT d'un signal réel
import scipy
import matplotlib.pyplot as plt
from math import sqrt

import simpleaudio as sa

def play(WAV_file):
    '''
    Plays the sound of the given WAV file.
    '''
    wave_obj = sa.WaveObject.from_wave_file(WAV_file)
    play_obj = wave_obj.play()
    play_obj.wait_done()  # Wait until sound has finished playing
    
    
    
def plot_sig_ech(t_ech, s_ech, 
                 title='Signal discrétisé', 
                 xlabel='Temps [s]',
                 ylabel='Signal [Unité arbitraire]'):
    '''
    Trace les barres verticales montrant l'amplitude du signal aux temps échantillonnés.
    
    Arguments:
      t_ech: le vecteur [t_0, t_1... t_N-1] des N instants d'échantillonnage du signal
             ou la période d'échantilonnage.
      s_ech: le vecteur [s(t_0), s(t_1)...] des valeurs du signal s aux temps échantillonnés
      title: titre du tracé (défaut: 'Signal discrétisé').
      xlabel: le label de l'axe du temps (defaut: 'Temps [s]')
      ylabel: le label de l'axe Y (défaut: 'Signal [Unité arbitraire]')
    '''
    if isinstance(t_ech, int) or isinstance(t_ech, float):
        Te = t_ech
        t_ech = np.arange(len(s_ech))*Te
    plt.figure(figsize=(8,4))
    markerline, stemlines, baseline = plt.stem(t_ech, s_ech, basefmt='C0')
    markerline.set_markerfacecolor('white')
    markerline.set_markersize(3.5)
    baseline.set_color('grey')
    baseline.set_linewidth(0.5)
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
def plot_spectre_amplitude(f_ech, spectre, f_max=None, 
                           title="Spectre d'amplitude", 
                           nb_line_printed=0, 
                           plot_harmonics=False):
    '''
    Trace les raies du spectre d'amplitude d'un signal discrétisé.
    
    Arguments:
                 f_ech: le vecteur des fréquences discrètes
               spectre: le vecteur des amplitude du spectre pour les fréquences discrètes.
                 title: titre du tracé (défaut: "Spectre d'amplitude").
      nb_line_printed: nombre de raies listés sous le graphe (fréquence et amplitude), défaut=0
        plot_harmonics: Pour faire tracer les raies harmoniques, if faut donner un couple 
                        (nbre max d'hamoniques, width) dont la signification est donnée 
                        avec la fonction find_harmonics. Valeur par défault: False. 
    '''
    plt.figure(figsize=(8,4))
    markerline, stemlines, baseline = plt.stem(f_ech, spectre, basefmt='C0')
    markerline.set_markerfacecolor('white')
    markerline.set_markersize(3.5)
    baseline.set_color('grey')
    baseline.set_linewidth(0.5)
    if f_max is None:
        f_max = f_ech[-1]
    plt.xlim(0, f_max)
    plt.grid()
    plt.title(title)
    plt.xlabel("F [Hz]")
    plt.ylabel("Amplitude [unité arbitraire]")
    
    if plot_harmonics:
        nb_harm, width = plot_harmonics
        index_harmo, freq_harmo, ampl_harmo  = find_harmonics(spectre, f_ech, nb_harm, width)
        plt.plot(freq_harmo, ampl_harmo, 'xr', label="Harmoniques détectées (ampl. pic)")
        
    #plt.legend()
    plt.show()
    
    if nb_line_printed:
        if nb_line_printed >= len(spectre): nb_line_printed=len(spectre)-1
        max_value, max_index = extract_maxima(spectre)
        print("Pics du spectre :")
        f1 = -1
        for i in max_index[:nb_line_printed]:
            f2 = f_ech[i]
            if f2 > f1 :
                print(f"\t{f2:10.1f} Hz\t{spectre[i]:8.4f}") 
                f1 = f2
            else:
                break
      
    if plot_harmonics:
        print("\tRang\t Fréquence\tAmpl. pic") 
        for i, (_, freq, ampl) in enumerate(zip(index_harmo, freq_harmo, ampl_harmo), 1):
            print(f"\t{i:4d}\t{freq:7.1f} Hz\t {ampl:8.4f}")
        return freq_harmo, ampl_harmo

                
def find_harmonics(spectre, freq, nb_harm=10, width=50, nb_pic_eff=5):
    '''
    Extraire les raies harmoniques d'un tableau ndarray de raies spectrales.
    Arguments:
       spectre:  le vecteur des amplitude des raies spectrales
          freq:  le vecteur des fréquences
       nb_harm:  nbre max d'harmoniques à traiter
         width:  nbre de points pris en compte de part et d'autre de la position théorique de 
                 l'harmonique pour trouver la position réelle de la raie la plus grande. 
    Renvoie: 
        le vecteur des indices des des raies harmoniques trouvées dans spectre
        le vecteur des fréquences des harmoniques
        le vecteur des amplitudes du pic principal des harmoniques.
    '''
    index_harmo, freq_harmo, ampl_harmo = [], [], []
    
    # l'idice de la raie du fondamental
    index = np.argmax(spectre)
    
    index_harmo.append(index)
    freq_harmo.append(freq[index])
    ampl_harmo.append(spectre[index])
    
    i = 1
    while True:
        # estimation de l'index du prochain pic dan sle spectre:
        index += index_harmo[0]
        fin = min(len(spectre), index+width)
        if index-width >= fin: break
        index = index - width + np.argmax(spectre[index-width:fin])

        index_harmo.append(index)
        freq_harmo.append(freq[index])
        ampl_harmo.append(spectre[index])

        i +=1
        if i > nb_harm : break
        
    return index_harmo, freq_harmo, ampl_harmo


def extract_maxima(arr):
    '''
    Extraire les maximums d'un tableau ndarray
    Arguments:
        arr: ndarray à traiter
    Renvoie: le vecteur des valeurs maximales trouvées dans arr
             le vecteur des indices des max dans le arr.
    '''
    buffer = np.array(arr).copy()
    max_value = []
    max_index = []
    index = np.argmax(buffer)
    while True:
        max_index.append(index)
        max_value.append(buffer[index])
        buffer[index] = -np.inf
        new_index = np.argmax(buffer)
        if new_index == index:
            break
        index = new_index
    return max_value, max_index


def process_periodic_signal(x, Fs, Fe, D, 
                            temporal_plot=True, 
                            nb_line_printed=0,
                            temporal_title="Tracé temporel",
                            spectral_title="Spectre d'amplitude"):
    '''
    - Calcule le vecteur des temps discrets,
    - Trace l'allure temporelle du signal discrétisé
    - Calcule la FFT et le spectre d'amplitude du signal,
    - Trace le spectre d'amplitude.
    
    Arguments :
      x : fonction vectorisée définissant le signal périodique, qui doit prendre 
          2 arguments : le vecteur des instants d'échantillonage, et la période du signal.
      Fs: fréquence du signal x en Hertz.
      Fe: fréquence d'échantilonnage en Hertz.
      D : durée de l'échantillonnage en secondes. 
      nb_line_printed: nombre de raies listées sous le graphe (fréquence et amplitude), défaut=0
      temporal_title: titre du tracé temporel
      spectral_title: titre du tracé du spectre
    
    Retour:
      renvoie t_ech, x_ech, f_ech, A
        t_ech: vecteur des instants d'échantillonnage
        x_ech: vecteur des valeur du signal échantillonné
        f_ech: vecteur des fréquences discrètes
        A: spectre d'amplitude du signal échantillonné
    '''
    
    Te, Ts = 1/Fe, 1/Fs
    
    t_ech = np.arange(0, D, Te)
    N = len(t_ech)
    x_ech = x(t_ech, Ts)            # calcul du signal pour les temps échantillonnés
    
    if temporal_plot:
        plot_sig_ech(t_ech, x_ech, temporal_title)

    X  = rfft(x_ech)          # FFT du signal => X est à avaleurs complexes
    A = 2*np.absolute(X)/N    # spectre d'amplitude = module de la FFT
    delta_f = Fe/N            # Fe divisé par le nombre de points
    
    f_ech = np.arange(0, len(X))*delta_f   # vecteurs des points en fréquence
    f_ech[-1]
    plot_spectre_amplitude(f_ech, A, nb_line_printed=nb_line_printed, title=spectral_title)
    
    print(f"le pas en fréquence est : {delta_f:8.3} Hz")
    
    i_max = A.argmax()
    f_max, A_max = f_ech[i_max], A[i_max]
    print(f"Valeur du pic du spectre d'amplitude à {f_max:.3f} Hz : {A_max:.3f}")
    
    return t_ech, x_ech, f_ech, A 
    
