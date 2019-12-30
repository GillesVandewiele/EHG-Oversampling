import numpy as np
import scipy

from ehgfeatures.features import EMDDecomposition, FeatureBase

__all__=['FeaturesSadiAhmed']

class FeaturesSadiAhmed(FeatureBase):
    """
    Based on:

    @article{article,
                author = {Sadi-Ahmed, Nafissa and Kacha, Baya and Taleb, Hamza and Kedir-Talha, Malika},
                year = {2017},
                month = {12},
                pages = {},
                title = {Relevant Features Selection for Automatic Prediction of Preterm Deliveries from Pregnancy ElectroHysterograhic (EHG) records},
                volume = {41},
                journal = {Journal of Medical Systems},
                doi = {10.1007/s10916-017-0847-8}
                }
    
    number of features according to paper: 7 features extracted from 2 EMD functions: 14
    """

    def __init__(self, imfs= [3, 6], sampling_frequency= 20.0):
        self.imfs= imfs
        self.sampling_frequency= sampling_frequency

    def n_features(self):
        return 7*len(self.imfs)

    def extract(self, signal):
        emds= EMDDecomposition(n_levels= max(self.imfs)+1).extract(signal)

        def split(array):
            left= 0.0
            right= np.sum(array)
            diffs= []
            for i in range(len(array)):
                left= left + array[i]
                right= right - array[i]
                diffs.append(abs(left - right))
            return np.argmin(diffs)
        
        results= {}

        for i in self.imfs:
            emd= emds['emd_' + str(i)]

            analytic_signal= scipy.signal.hilbert(emd)

            amplitude_envelope= np.abs(analytic_signal)

            instantaneous_phase= np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency= (np.diff(instantaneous_phase)/(2.0*np.pi)*self.sampling_frequency)

            perio= scipy.signal.periodogram(emd, fs= self.sampling_frequency)

            vertical_threshold= 0.01*(np.max(instantaneous_frequency) - np.min(instantaneous_frequency))
            horizontal_threshold= np.max([int(0.001*len(instantaneous_frequency)), 1])

            n_peaks= len(scipy.signal.find_peaks(np.abs(instantaneous_frequency), threshold= vertical_threshold, distance=horizontal_threshold)[0])

            med_freq= perio[0][split(perio[1])]

            fwl_mask= np.logical_and(perio[0] > 0.1, perio[0] <= 0.2)
            fwl= (perio[0][fwl_mask], perio[1][fwl_mask])

            fwh_mask= np.logical_and(perio[0] > 0.2, perio[0] <= 3)
            fwh= (perio[0][fwh_mask], perio[1][fwh_mask])

            fwh_peak_freq= fwh[0][np.argmax(fwh[1])]
            fwl_peak_freq= fwl[0][np.argmax(fwl[1])]
            gap_freq= fwh_peak_freq - fwl_peak_freq
            fwh_peak_power= np.max(fwh[1])
            fwl_peak_power= np.max(fwl[1])
            gap_power= fwl_peak_power - fwh_peak_power

            prefix= self.__class__.__name__ + '_emd_' + str(i) + '_'
            results[prefix + 'n_peaks']= n_peaks
            results[prefix + 'med_freq']= med_freq
            results[prefix + 'fwh_peak_freq']= fwh_peak_freq
            #results[prefix + 'fwl_peak_freq']= fwl_peak_freq
            results[prefix + 'gap_freq']= gap_freq
            results[prefix + 'fwh_peak_power']= fwh_peak_power
            results[prefix + 'fwl_peak_power']= fwl_peak_power
            results[prefix + 'gap_power']= gap_power
        
        return results