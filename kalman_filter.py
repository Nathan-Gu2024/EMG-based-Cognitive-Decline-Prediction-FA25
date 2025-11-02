import numpy as np



class KalmanFilter: 
    transition, control, observation, proc_noise, measure_noise, estimate0, error_cov0 = None, None, None, None, None, None, None
    # F: State transition matrix (system model).
    # B: Control matrix (effect of control input).
    # H: Observation matrix (how we measure the state).
    # Q: Process noise covariance (uncertainty in the process).
    # R: Measurement noise covariance (uncertainty in the measurements).
    # x0: Initial state estimate.
    # P0: Initial error covariance (initial uncertainty of state estimate).
    def __init__(self, F, B, H, Q, R, x0, P0):
        self.transition = F
        self.control = B
        self.observation = H
        self.proc_noise = Q
        self.measure_noise = R
        self.estimate0 = x0
        self.error_cov0 = P0
    
    def predict(self, u):
        self.estimate0 = np.dot(self.transition, self.estimate0) + np.dot(self.control, u)
        self.error_cov0 = np.dot(self.transition, np.dot(self.P, self.transition.T)) + self.proc_noise
        return self.estimate0
    
    def update(self, z):
        S = np.dot(self.observation, np.dot(self.error_cov0, self.observation.T)) + self.measure_noise
        K = np.dot(np.dot(self.error_cov0, self.observation.T), np.linalg.inv(S))
        y = z - np.dot(self.observation, self.estimate0)
        self.estimate0 = self.estimate0 + np.dot(K, y)
        I = np.eye(self.error_cov0.shape[0])
        self.error_cov0 = np.dot(I - np.dot(K, self.observation), self.error_cov0)
        return self.estimate0
