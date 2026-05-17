clear; clc; close all;

%% 1. Inizializzazione del Robot % Carichiamo il modello del Panda 
% "Panda" è il nome comune del robot frankaEmikaPanda: un braccio robotico
% collaborativo a 7 giunti prodotto da Franka Emika, spesso usato in ricerca.
panda = loadrobot("frankaEmikaPanda", "DataFormat", "column"); %column indica i dati in vett. colonna
ee_name = 'panda_hand'; % Nome del frame dell'end-effector

%% 2. Parametri di Simulazione e Traiettoria 
dt = 0.005; % Passo di integrazione molto fine per la stabilità in 3D 
T_traj = 1.0; % La traiettoria deve essere eseguita in tot secondi

r = 0.4; % Raggio del cerchio 
omega = 2*pi / T_traj;

% Guadagni del controllore (ora abbiamo 6 gradi di libertà nel task: 3 rot, 3 trasl) 
K = 10 * eye(6);

%% 3. Configurazione Iniziale 
q = [0; -pi/4; 0; -3*pi/4; 0; pi/2; pi/4]; 
q_9 = [q; 0; 0];
T_init = getTransform(panda, q_9, ee_name); 
p_init = T_init(1:3, 4); 
R_des = T_init(1:3, 1:3);

center = [0.2; 0.2; 0.5];

% Definiamo il punto di partenza: dove si trova il cerchio a t=0
P_start = [center(1) - r; center(2); center(3)];

%% 4. FASE 1: Traiettoria di Allineamento
% Calcoliamo quanti step servono basandoci sulla distanza
distanza = norm(P_start - p_init);
tempo_assestamento = distanza / 0.1; % ipotizziamo una convergenza a 0.1 m/s
t_ass = 0:dt*10:tempo_assestamento; %*10 perche arrivava molto prima del previsto, velocità non costante
N_ass = length(t_ass);

errors_ass = [];

for k = 1:N_ass
    q_9 = [q; 0; 0];
    
    T_curr = getTransform(panda, q_9, ee_name);
    p_curr = T_curr(1:3, 4);
    R_curr = T_curr(1:3, 1:3);
    
    % L'errore punta verso l'inizio del cerchio (P_start), non a zero!
    e_p = P_start - p_curr;
    e_o = 0.5 * ( cross(R_curr(:,1), R_des(:,1)) + cross(R_curr(:,2), R_des(:,2)) + cross(R_curr(:,3), R_des(:,3)) );
    e = [e_o; e_p]; 

    errors_ass = [errors_ass, e_p];

    J_full = geometricJacobian(panda, q_9, ee_name);
    J = J_full(:, 1:7); 
    
    % La velocità di feedforward è zero, ci muoviamo solo per annullare l'errore
    v_task = zeros(6,1) + K * e;
    
    q_dot = pinv(J) * v_task;
    q = q + q_dot * dt;

    if mod(k, 5) == 0
        show(panda, [q; 0; 0], 'Frames', 'off', 'PreservePlot', false);
        drawnow;
    end
end

%% 5. Generazione della Traiettoria Circolare (OFFLINE) 
t = 0:dt:T_traj; 
N = length(t); 

Pd_dot = zeros(6, N); 
P_des = zeros(3, N);

for k = 1:N 
    P_des(1, k) = center(1) - r * cos(omega * t(k)); 
    P_des(2, k) = center(2) + r * sin(omega * t(k)); 
    P_des(3, k) = center(3); 
    
    Pd_dot(4, k) =  r * omega * sin(omega * t(k));
    Pd_dot(5, k) =  r * omega * cos(omega * t(k));
    Pd_dot(6, k) =  0;
end
    
%% 6. Setup Grafico 
fig = figure; 
show(panda, [q;0;0], 'Frames', 'off', 'PreservePlot', false); 
hold on; grid on; light; view(3); 
plot3(P_des(1,:), P_des(2,:), P_des(3,:), 'k--', 'LineWidth', 1.5);
    
errors = [];

%% 7. FASE 2: Loop di Controllo CLIK per il cerchio
for k = 1:N
    
    if ~isgraphics(fig)
        disp('Hai chiuso la finestra. Simulazione interrotta!');
        break; 
    end
    
    q_9 = [q; 0; 0];
    
    T_curr = getTransform(panda, q_9, ee_name);
    p_curr = T_curr(1:3, 4);
    R_curr = T_curr(1:3, 1:3);
    
    % Indici puliti: usiamo direttamente k
    e_p = P_des(:, k) - p_curr;
    e_o = 0.5 * ( cross(R_curr(:,1), R_des(:,1)) + cross(R_curr(:,2), R_des(:,2)) + cross(R_curr(:,3), R_des(:,3)) );
    e = [e_o; e_p]; 
    errors = [errors, e_p]; 
    
    J_full = geometricJacobian(panda, q_9, ee_name);
    J = J_full(:, 1:7); 
    
    v_task = Pd_dot(:, k) + K * e;
    
    J_pinv = pinv(J);
    q_dot_0 = zeros(7, 1); 
    q_dot = J_pinv * v_task + (eye(7) - J_pinv * J) * q_dot_0;
    
    q = q + q_dot * dt;
    
    if mod(k, 5) == 0
        show(panda, [q; 0; 0], 'Frames', 'off', 'PreservePlot', false);
        drawnow;
    end
end

%% Plot errori
figure;
plot(t_ass, errors_ass); hold on;
grid on;
legend("error x", "error y", "error z");
title("Errori fase allinemento");

figure;
plot(t, errors); hold on;
grid on;
legend("error x", "error y", "error z");
title("Errori follow traiettoria");