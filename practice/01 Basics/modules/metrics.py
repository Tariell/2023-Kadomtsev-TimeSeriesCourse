import numpy as np


def ED_distance(ts1, ts2):
    """
    Calculate the Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    ed_dist : float
        Euclidean distance between ts1 and ts2.
    """
    
    if len(ts1) != len(ts2):
        raise ValueError("The two arrays must have the same length.")
    
    ed_dist = np.sqrt(np.sum((ts1 - ts2)**2))

    return ed_dist


def norm_ED_distance(ts1, ts2):
    """
    Calculate the normalized Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    norm_ed_dist : float
        The normalized Euclidean distance between ts1 and ts2.
    """

    if len(ts1) != len(ts2):
        raise ValueError("The two arrays must have the same length.")

    m = len(ts1)
    
    norm_ed_dist = np.sqrt(abs(2*m* (1 - (np.dot(ts1,ts2) - m * (sum(ts1)/m) * (sum(ts2)/m)) / (m * np.sqrt( sum(ts1**2 - (sum(ts1)/m)**2) / m ) * np.sqrt( sum(ts2**2 - (sum(ts2)/m)**2) / m ) ))))
    
    return norm_ed_dist


def DTW_distance(ts1, ts2, r=None):
    """
    Calculate DTW distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    r : float
        Warping window size.
    
    Returns
    -------
    dtw_dist : float
        DTW distance between ts1 and ts2.
    """
    
    # Инициализация матрицы расстояний
    n = len(ts1)
    m = len(ts2)
    dtw_matrix = np.zeros((n+1, m+1))
    
    # Заполнение первой строки и первого столбца бесконечными значениями
    dtw_matrix[0, :] = np.inf
    dtw_matrix[:, 0] = np.inf
    dtw_matrix[0, 0] = 0
    
    # Вычисление DTW меры
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.square(ts1[i-1] - ts2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])

    dtw_dist = dtw_matrix[n, m]

    return dtw_dist