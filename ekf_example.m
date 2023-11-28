% Time settings
T = 10;            % Total simulation time in seconds
dt = 0.1;          % Time step (Delta t)
time = 0:dt:T;     % Time vector

% State-Space model (assuming linear state transition for simplicity)
A = [1 dt; 0 1];   % State transition matrix
Q = 0.01 * eye(2); % Process noise covariance

% Initial State
x_initial = [0; 0]; % Initial angular position and velocity

% Initialization
x_est = zeros(2, length(time));    % State estimate initialization
x_est(:, 1) = x_initial;            % Set initial state
P_est = zeros(2, 2, length(time)); % Error covariance initialization
P_est(:, :, 1) = eye(2);           % Initial error covariance

% Simulate some true states (for demonstration purposes)
theta_true = sin(0.1*time);          % True angle (sine wave)
omega_true = 0.1*cos(0.1*time);      % True angular velocity
x_true = [theta_true; omega_true];   % True state matrix

% Creating noisy measurements
R = 0.1;           % Measurement noise covariance
z = sin(theta_true) + sqrt(R)*randn(size(theta_true)); % Noisy measurements

% Nonlinear Measurement Function
h = @(x) sin(x(1));

% EKF Implementation
for k = 2:length(time)
    % State Prediction
    x_pred = A * x_est(:, k-1);
    P_pred = A * P_est(:, :, k-1) * A' + Q;

    % Linearize Measurement Function (Jacobian)
    H_jacobian = [cos(x_pred(1)) 0]; % Partial derivatives of h

    % Measurement Update
    y_k = z(k) - h(x_pred); % Innovation
    S = H_jacobian * P_pred * H_jacobian' + R; % Residual covariance
    K = P_pred * H_jacobian' / S; % Kalman Gain
    x_est(:, k) = x_pred + K * y_k;
    P_est(:, :, k) = (eye(2) - K * H_jacobian) * P_pred;
end

% Plotting the results
figure;
subplot(2,1,1);
plot(time, theta_true, 'g', 'DisplayName', 'True Angle');
hold on;
plot(time, z, 'b', 'DisplayName', 'Noisy Measurements');
plot(time, x_est(1, :), 'r', 'DisplayName', 'EKF Estimate of Angle');
xlabel('Time (s)');
ylabel('Angle (rad)');
title('EKF for Robot Arm Angle Estimation');
legend;

subplot(2,1,2);
plot(time, omega_true, 'g', 'DisplayName', 'True Angular Velocity');
hold on;
plot(time, x_est(2, :), 'r', 'DisplayName', 'EKF Estimate of Angular Velocity');
xlabel('Time (s)');
ylabel('Angular Velocity (rad/s)');
title('EKF Angular Velocity Estimation');
legend;
