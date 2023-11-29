clear, clc 
close all 
% Time settings
T = 10;            % Total simulation time in seconds
dt = 0.1;          % Time step (Delta t)
time = 0:dt:T;     % Time vector

% Initial State
x_initial = [0; 0]; % Initial angular position and velocity

% Initialization
x_est = zeros(2, length(time));    % State estimate initialization
x_est(:, 1) = x_initial;            % Set initial state
P_est = zeros(2, 2, length(time)); % Error covariance initialization
P_est(:, :, 1) = eye(2);           % Initial error covariance

% Process and Measurement Noise Covariances
Q = 0.01 * eye(2); % Process noise covariance
R = 0.1;           % Measurement noise covariance

% Simulated true states and noisy measurements (for demonstration)
theta_true = sin(0.1*time); % True angle
omega_true = 0.1*cos(0.1*time); % True angular velocity
z = sin(theta_true) + sqrt(R)*randn(size(theta_true)); % Noisy measurements

% Nonlinear State Transition Function
f = @(x) [x(1) + dt*x(2); x(2) + dt*sin(x(1))];

% Nonlinear Measurement Function
h = @(x) sin(x(1));

% EKF Implementation
for k = 2:length(time)
    % State Prediction
    x_pred = f(x_est(:, k-1));

    % Jacobian of f at x_pred
    F_jacobian = [1 dt; dt*cos(x_pred(1)) 1];

    % Predict Error Covariance
    P_pred = F_jacobian * P_est(:, :, k-1) * F_jacobian' + Q;

    % Linearize Measurement Function (Jacobian)
    H_jacobian = [cos(x_pred(1)) 0];

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

