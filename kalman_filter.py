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
        self.error_cov0 = np.dot(self.transition, np.dot(self.error_cov0, self.transition.T)) + self.proc_noise
        return self.estimate0
    
    def update(self, z):
        S = np.dot(self.observation, np.dot(self.error_cov0, self.observation.T)) + self.measure_noise
        K = np.dot(np.dot(self.error_cov0, self.observation.T), np.linalg.inv(S))
        y = z - np.dot(self.observation, self.estimate0)
        self.estimate0 = self.estimate0 + np.dot(K, y)
        I = np.eye(self.error_cov0.shape[0])
        self.error_cov0 = np.dot(I - np.dot(K, self.observation), self.error_cov0)
        return self.estimate0
    
    #Creates the matrices for the kalman
    #Generated with GPT because I have no clue what id put as input
    def create_kalman_filter(self, dt=0.002):
        # State: [angle, bias]
        F = np.array([[1, -dt],
                    [0,  1]])          # State transition (angle decreases by bias each step)

        B = np.array([[dt],
                    [0]])              # Control input (gyro rate)

        H = np.array([[1, 0]])           # We measure only the angle (from accelerometer)

        Q = np.array([[1e-5, 0],
                    [0, 1e-6]])        # Process noise covariance
        R = np.array([[1e-2]])           # Measurement noise covariance

        x0 = np.zeros((2, 1))            # Initial state [angle=0, bias=0]
        P0 = np.eye(2) * 0.01            # Initial covariance

        return KalmanFilter(F, B, H, Q, R, x0, P0)


