import os
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d
from SoundSurrogate import SoundSurrogate
import utils
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.backend.set_floatx('float64')

class TurbSound():
    def __init__(self, model_path=None):
        '''
            Parameters:
                model          - neural network surrgoate model for 
                                 sound predictions at fixed observer locations
                sf             - scaling factor for model inputs and output
                N_in           - input dimension
                N_obs          - number of observers
                r, theta       - polar coordinates of observers
                x, y           - Cartesian coordinates of observers
                default_values - dictionary of default atmospheric values
        '''
        super(TurbSound, self).__init__()

        # Define observer locations for model output
        r = np.concatenate((np.arange( 10.,  500., 10.), 
                            np.arange(500., 5001., 20.)), axis=0)
        theta = np.arange(0., 360., 22.5)
        self.r, self.theta = np.meshgrid(r, theta)
        self.x = self.r*np.cos(-(np.pi/180.)*self.theta + np.pi/2)
        self.y = self.r*np.sin(-(np.pi/180.)*self.theta + np.pi/2)
        self.N_obs = self.r.size
        
        # Create dictionary of default environmental variables
        self.defaults = {'wind_speed': 10.,
                     'wind_direction': 0.,
                     'turb_intensity': 0.2,
                    'air_temperature': 10.,
                       'rel_humidity': 70.,
                       'air_pressure': 103.5,
                      'ground_factor': 0.5}
        self.N_in = len(self.defaults)

        # Initialize trained model
        self.model = SoundSurrogate(tf.keras.Input(shape=(self.N_in-1,)),
                                    tf.keras.Input(shape=(self.N_obs,)), 
                                    model_path=model_path)

        # Set scaling factors for model inputs and outputs
        self.sf = {'x': np.array([[ 4., 25.], [0.15, 0.4], [-10., 35.],
                                  [10., 95.], [80., 105.], [  0., 0.5]]),
                   'f': 100.}

    def _apply_rotation_(self, f, wd):
        rot_steps = np.round(wd/22.5).astype(int)
        f = np.array([np.roll(f_i, r_i, axis=0) for f_i, r_i in zip(f, rot_steps)])
        
        return f

    def compute_sound(self, **kwargs):
        '''
            Given input environmental conditions, computes the sound 
            level at the N_obs oberservers.

            Input parameters can be provided in the following formats:
                x - Dictionary with any number of the following keys:
                      'wind_speed'
                      'wind_direction'
                      'turbulence_intensity'
                      'air_temperature'
                      'relative_humidity'
                      'air_pressure'
                      'ground_factor'
                    Value provided for each key can by a single value or
                    N_samples in a list or NumPy array. Any keys not 
                    provided automatically use default values.
                x - (N_samples x 7) NumPy array where 6 columns correspond
                    to the inputs listed in the order above.

            Additional flags:
                normalize    - (bool)[default: True] Flag for if input 
                               data needs to be normalized.
                batch_size   - (int)[default: 100] Number of samples per
                               batch when evaluating the model.
                check_bounds - (bool)[default: True] Flag to determine
                               if the input ranges need to be checked.
        '''
        keys = [_.lower() for _ in kwargs.keys()]

        # Determine if input ranges need to be checked or not
        check_bounds = kwargs['check_bounds'] if 'check_bounds' in kwargs.keys() else True
        
        # Processing input data to extract wind direction and get standard array format.
        X, wd = utils._process_atmospheric_inputs_(kwargs, self.defaults, check_bounds)
        
        # Normalize input data if required
        if ('normalize' not in kwargs.keys()) or \
           (('normalize' in kwargs.keys()) and (kwargs['normalize'])):
            X = utils.norm_input_data(X, self.sf['x'])
        
        # Set the batch size for model evalutions
        batch_size = kwargs['batch_size'] if 'batch_size' in kwargs.keys() else 100
        
        # Evaluate the model and collect the unnormalized sound levels
        X = tf.data.Dataset.from_tensor_slices(X.T).batch(batch_size)
        f = np.zeros((0, self.N_obs))
        for X_i in X:
            f_i = utils.unnorm_output_data(self.model.run_model(X_i), self.sf['f'])
            
            f = np.concatenate((f, f_i), axis=0)
        f = f.reshape((f.shape[0], self.r.shape[0], self.r.shape[1]))
        f = np.roll(np.flip(f, axis=1), 9, axis=1)

        # Rotate sound level based on wind direciton
        f = self._apply_rotation_(f, wd)

        return f

    def sound_metrics(self, metric_list=['Leq'], **kwargs):#, worst_case=False
        '''
            Given a series of input environmental conditions, compute specified 
            sound metrics at each oberserver location.

            metric_list   - ['Leq', 'Lmax', 'L{percentile}' (e.g., 'L90')]
                            List of sound metrics to compute.

            All other inputs match self.compute_sound()
        '''
        # Gather and process flags
        metric_list = [metric_list] if isinstance(metric_list, str) else metric_list
        normalize = kwargs['normalize'] if 'normalize' in kwargs.keys() else True
        batch_size = kwargs['batch_size'] if 'batch_size' in kwargs.keys() else 100
        check_bounds = kwargs['check_bounds'] if 'check_bounds' in kwargs.keys() else True

        # Processing input data to extract wind direction and get standard array format.
        X, wd = utils._process_atmospheric_inputs_(kwargs, self.defaults, check_bounds)
        X = np.concatenate((X[:1, :], wd.reshape((1, -1)), X[1:, :]), axis=0)
        N_samples = X.shape[1]

        if batch_size == -1:
            batch_size = N_samples

        X = tf.data.Dataset.from_tensor_slices(X.T).batch(batch_size)

        # Evaluate the model for all inputs while aggregating outputs
        # in an efficient manner to compute sound statistics.
        L_min = 1e8*np.ones((self.N_obs, ))
        L_max = np.zeros((self.N_obs, ))
        L_eq = np.zeros((self.N_obs, ))
        L_bins = np.linspace(-1., 100., 102)
        L_counts = np.zeros((L_bins.size-1, self.N_obs), dtype=int)
        for X_i in X:
            f = self.compute_sound(x=X_i.numpy(), normalize=normalize, batch_size=batch_size, check_bounds=False)
            f = f.reshape((X_i.shape[0], self.N_obs))

            L_min = np.minimum(L_min, np.min(f, axis=0))
            L_max = np.maximum(L_max, np.max(f, axis=0))

            L_eq = L_eq + np.sum((10.**(f/20.))**2., axis=0)

            L_counts = L_counts + np.apply_along_axis(lambda a: np.histogram(a, bins=L_bins)[0], 0, f)
        L_bins = 0.5*(L_bins[:-1]+L_bins[1:])
        L_counts = np.cumsum(L_counts, axis=0)/np.sum(L_counts, axis=0)

        # Convert the aggregated sound values into the specified metrics.
        L_metric = []
        for st in metric_list:
            if st == 'Lmax':
                L_metric.append(L_max)
            elif st == 'Leq':
                L_metric.append(10.*np.log10(L_eq/float(N_samples)))
            else: # 'L{percentile}'
                p = float(st[1:])
                assert (p >= 0.) and (p <= 100.), 'Percentile must be between 0 and 100.'
                Lp = np.zeros((self.N_obs, ))
                for i in range(self.N_obs):
                    L_counts_i = L_counts[:, i]

                    idx0 = np.where(L_counts_i == 0.)[0]
                    idx0 = idx0[-1] if len(idx0) > 0 else 0
                    idx1 = np.where(L_counts_i == 1.)[0]
                    idx1 = idx1[0]+1 if len(idx1) > 0 else L_counts_i.size

                    L_counts_i = L_counts_i[idx0:idx1]
                    L_bins_i = L_bins[idx0:idx1]
                    L_bins_i[0] = L_min[i]
                    L_bins_i[-1] = L_max[i]

                    f = interp1d(L_counts_i, L_bins_i)
                    Lp[i] = f((1. - p/100.))
                L_metric.append(Lp)
        L_metric = [L.reshape((self.r.shape[0], self.r.shape[1])) for L in L_metric]
        L_metric = L_metric[0] if len(L_metric) == 1 else L_metric
        
        return L_metric

    def sound_setback(self, f, thresh, worst_case=False):
        '''
            Given sound data at each oberserver location and a sound threshold,
            compute the setback distance for either .

            f          - (Numpy array) Sound levels for every observer.
            thresh     - (float) Decibel threshold level.
            worst_case - (bool)[default: False] Flag for whether to calculate
                         setbacks using the worst case approach or directional.
        '''

        if worst_case:
            # For worst_case analysis, find loudest sound level at each radial distance
            if f.ndim == 2:
                f = np.max(f, axis=0)
            tf = (f >= thresh)

            # Compute 1d setback distance
            setback_dist = np.max(self.r[0, tf]) if np.sum(tf) > 0. else 0.
        else:
            # For directional analysis, check that sound levels are 2d
            assert f.ndim == 2
            tf = (f >= thresh)

            # Compute setback distance for each angular direction
            setback_dist = np.zeros((self.theta.shape[0], ))
            for i in range(self.theta.shape[0]):
                setback_dist[i] = np.max(self.r[i, tf[i, :]]) if np.sum(tf[i, :]) > 0. else 0.

        return setback_dist









