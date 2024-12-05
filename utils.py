import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_input_data(filepath, bounds=None, norm_data=True):
    if bounds is None:
        bounds = [[4., 25.],     # Wind Speed
                  [0.15, 0.4],   # Turbulence Intensity
                  [-10., 35.],   # Air Temperature
                  [10., 95.],    # Relative Humidty
                  [80., 105.],   # Air Pressure
                  [0., 0.5]]     # Ground Factor
        bounds = np.array(bounds)

    columns = ['wind_speed', 'turb_intensity', 'air_temp', 
               'rel_humidity', 'air_pressure', 'ground_factor']
    df = pd.read_csv(filepath)
    X = df[columns].to_numpy()

    if norm_data:
        X = norm_input_data(X, bounds)

    return X.astype(np.float32)

def load_output_data(filepath, sf=100., norm_data=True):
    f = np.load(filepath)[2, ...]
    f[f < 0.] = 0.

    if norm_data:
        f = norm_output_data(f, sf)

    f = np.transpose(f, [2, 0, 1])
    f = f.reshape((f.shape[0], -1))

    return f.astype(np.float32)

def norm_input_data(X, bounds):
    X = np.atleast_2d(X)
    X = (2/(bounds[:, 1:] - bounds[:, :1]))*(X - bounds[:, :1]) - 1.

    return X

def unnorm_input_data(X, bounds):
    X = np.atleast_2d(X)
    X = ((bounds[:, 1:] - bounds[:, :1])/2)*(X + 1.) + bounds[:, :1]

    return X

def norm_output_data(f, sf):
    f /= sf

    return f

def unnorm_output_data(f, sf):
    f *= sf

    return f

def _check_input_bounds_(x, bounds=None):
    if bounds is None:
        bounds = [[4., 25.],     # Wind Speed w/ extra weighting [4, 15]
                  [0., 360.],     # Wind Speed w/ extra weighting [4, 15]
                  [0.15, 0.4],   # Mean T.I. w/ slope of 0.2/140
                  [-10., 35.],   # Air Temperature
                  [10., 95.],    # Relative Humidty
                  [80., 105.],   # Air Pressure
                  [0., 0.5]]     # Ground Factor
        bounds = np.array(bounds)

    tf_boundsOK = np.all((x >= bounds[:, :1]) & (x <= bounds[:, 1:]), axis=0)
    if np.any(np.logical_not(tf_boundsOK)):
        print('')
        print('WARNING: some samples provided are outside expected ranges.')    
        print('')

def _find_key_index_(k0, k1):
    idx = [i for i, k in enumerate(k0) if (k in k1)]
    idx.append(None)

    return idx[0]

def _parse_input_dict_(args, defaults=None):
    keys = [_.lower() for _ in args.keys()]

    possible_keys = ['wind_speed', 'speed', 'ws', 'u']
    idx = _find_key_index_(possible_keys, keys)
    if idx is not None:
        ws = np.atleast_2d(np.array(args[possible_keys[idx]]))
    else:
        ws = np.atleast_2d(np.array(defaults['wind_speed']))

    possible_keys = ['wind_direction', 'direction', 'wind_dir', 
                     'dir', 'ws', 'theta']
    idx = _find_key_index_(possible_keys, keys)
    if idx is not None:
        wd = np.atleast_2d(np.array(args[possible_keys[idx]]))
    else:
        wd = np.atleast_2d(np.array(defaults['wind_direction']))

    possible_keys = ['turbulence_intensity', 'turbulent_intensity'
                     'turb_intensity', 'turb_intense', 'ti']
    idx = _find_key_index_(possible_keys, keys)
    if idx is not None:
        ti = np.atleast_2d(np.array(args[possible_keys[idx]]))
    else:
        ti = np.atleast_2d(np.array(defaults['turb_intensity']))

    possible_keys = ['air_temperature', 'air_temp', 't']
    idx = _find_key_index_(possible_keys, keys)
    if idx is not None:
        T = np.atleast_2d(np.array(args[possible_keys[idx]]))
    else:
        T = np.atleast_2d(np.array(defaults['air_temperature']))

    possible_keys = ['relative_humidity', 'rel_humidity', 
                     'relative_humid', 'rel_humid', 'rh', 'h']
    idx = _find_key_index_(possible_keys, keys)
    if idx is not None:
        H = np.atleast_2d(np.array(args[possible_keys[idx]]))
    else:
        H = np.atleast_2d(np.array(defaults['rel_humidity']))

    possible_keys = ['air_pressure', 'air_press', 'pressure', 'p']
    idx = _find_key_index_(possible_keys, keys)
    if idx is not None:
        P = np.atleast_2d(np.array(args[possible_keys[idx]]))
    else:
        P = np.atleast_2d(np.array(defaults['air_pressure']))

    possible_keys = ['ground_factor', 'ground_fact', 'gf']
    idx = _find_key_index_(possible_keys, keys)
    if idx is not None:
        GF = np.atleast_2d(np.array(args[possible_keys[idx]]))
    else:
        GF = np.atleast_2d(np.array(defaults['ground_factor']))

    N_samples = np.max([ws.shape[1], ti.shape[1],  T.shape[1], 
                         H.shape[1],  P.shape[1], GF.shape[1]])
    for val in [ws, ti, T, H, P, GF]:
        assert (val.shape[1] == N_samples) or (val.shape[1] == 1)
    ws = np.repeat(ws, N_samples, axis=1) if (N_samples > ws.shape[1]) else ws
    wd = np.repeat(wd, N_samples, axis=1) if (N_samples > wd.shape[1]) else wd
    ti = np.repeat(ti, N_samples, axis=1) if (N_samples > ti.shape[1]) else ti
    T  = np.repeat(T,  N_samples, axis=1) if (N_samples >  T.shape[1]) else T
    H  = np.repeat(H,  N_samples, axis=1) if (N_samples >  H.shape[1]) else H
    P  = np.repeat(P,  N_samples, axis=1) if (N_samples >  P.shape[1]) else P
    GF = np.repeat(GF, N_samples, axis=1) if (N_samples > GF.shape[1]) else GF
    
    x = np.concatenate((ws, wd, ti, T, H, P, GF), axis=0)
    
    return x

def _process_atmospheric_inputs_(args, defaults=None, check_bounds=True):
    keys = [_.lower() for _ in args.keys()]
    if 'x' in keys:
        if isinstance(args['x'], dict):
            x = _parse_input_dict_(args['x'], defaults=defaults)
        elif isinstance(args['x'], np.ndarray):
            x = np.atleast_2d(args['x'])
            assert (x.shape[0] == 7) or (x.shape[1] == 7)
            if (x.shape[0] != 7) and (x.shape[1] == 7):
                x = x.T
        else:
            assert False, 'Bad input parameter "x" provided.'
    else:
        x = _parse_input_dict_(args, defaults=defaults)

    if check_bounds:
        _check_input_bounds_(x)

    x, wd = x[[0, 2, 3, 4, 5, 6], :], x[1, :]
    
    return x, wd
