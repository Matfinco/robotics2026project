clear; clc; close all;

%% 1. Inizializzazione del Robot % Carichiamo il modello del Panda 
% "Panda" è il nome comune del robot frankaEmikaPanda: un braccio robotico
% collaborativo a 7 giunti prodotto da Franka Emika, spesso usato in ricerca.
panda = loadrobot("frankaEmikaPanda", "DataFormat", "column"); %column indica i dati in vett. colonna
ee_name = 'panda_hand'; % Nome del frame dell'end-effector

%% 2. Parametri di Simulazione e Traiettoria 
dt = 0.005; % Passo di integrazione molto fine per la stabilità in 3D 
T_traj = 1.0; % La traiettoria deve essere eseguita in tot secondi
t = 0:dt:T_traj;
N = length(t);

r = 0.1; % Raggio del cerchio 
omega = 2*pi / T_traj;

% Guadagni del controllore (ora abbiamo 6 gradi di libertà nel task: 3 rot, 3 trasl) 
K = 20 * eye(6);

%% 3. Configurazione Iniziale 
q = [0; -pi/4; 0; -3*pi/4; 0; pi/2; pi/4]; % I nostri 7 giunti di controllo

% Creiamo una versione "paddata" a 9 giunti aggiungendo due zeri per la pinza chiusa 
q_9 = [q; 0; 0];

% Usiamo q_9 per interrogare MATLAB 
T_init = getTransform(panda, q_9, ee_name); 
p_init = T_init(1:3, 4); 
R_des = T_init(1:3, 1:3);

% Definiamo il centro del cerchio (cerchio orizzontale rispetto alla partenza) 
center = p_init + [r; 0; 0];

%% 4. Generazione della Traiettoria (OFFLINE) 
% Pre-allocazione per un task a 6 gradi di libertà (Pose) 
Pd_dot = zeros(6, N); % Prime 3: vel angolare, Ultime 3: vel lineare 
P_des = zeros(3, N);

for k = 1:N % Posizione desiderata (Cerchio orizzontale sul piano XY locale) 
    P_des(1, k) = center(1) - r * cos(omega * t(k)); 
    P_des(2, k) = center(2) + r * sin(omega * t(k)); 
    P_des(3, k) = center(3); % Altezza costante

    % Velocità lineare desiderata
    Pd_dot(4, k) =  r * omega * sin(omega * t(k));
    Pd_dot(5, k) =  r * omega * cos(omega * t(k));
    Pd_dot(6, k) =  0;
    
    % Le velocità angolari desiderate (Pd_dot 1,2,3) restano 0 
    % perché l'orientamento è costante.
end
    
    %% 5. Setup Grafico 
    fig = figure; % <-- SALVA L'HANDLE QUI! ASSEGNALO A 'fig' 
    show(panda, q_9, 'Frames', 'off', 'PreservePlot', false); 
    hold on; 
    grid on; 
    light; 
    view(3); 
    plot3(P_des(1,:), ...
        P_des(2,:), ...
        P_des(3,:), 'k--', 'LineWidth', 1.5);
    
    %% 6. Loop di Controllo CLIK 
 for k = 1:N
    
    % --- CONTROLLO ESISTENZA FINESTRA ---
    if ~isgraphics(fig)
        disp('Hai chiuso la finestra. Simulazione interrotta!');
        break; % Questo comando distrugge immediatamente il ciclo for
    end
    % ------------------------------------
    
    % 1. Padding per accontentare MATLAB
    q_9 = [q; 0; 0];
    
    % Cinematica Diretta
    T_curr = getTransform(panda, q_9, ee_name);
    p_curr = T_curr(1:3, 4);
    R_curr = T_curr(1:3, 1:3);
    
    % Calcolo dell'errore
    e_p = P_des(:, k) - p_curr;
    e_o = 0.5 * ( cross(R_curr(:,1), R_des(:,1)) + ...
                  cross(R_curr(:,2), R_des(:,2)) + ...
                  cross(R_curr(:,3), R_des(:,3)) );
    e = [e_o; e_p]; 
    
    % 2. Calcolo e "Taglio" dello Jacobiano
    J_full = geometricJacobian(panda, q_9, ee_name);
    J = J_full(:, 1:7); % <--- CRITICO: Estraiamo la sottomatrice 6x7 del solo braccio!
    
    % Legge di controllo per il task primario
    v_task = Pd_dot(:, k) + K * e;
    
    % Calcolo della Pseudoinversa (ora J è 6x7, quindi J_pinv sarà 7x6)
    J_pinv = pinv(J);
    
    % Null-space
    q_dot_0 = zeros(7, 1); 
    
    % Equazione completa (tutto è coerentemente a dimensione 7)
    q_dot = J_pinv * v_task + (eye(7) - J_pinv * J) * q_dot_0;
    
    % Integrazione (aggiorniamo solo i 7 giunti reali)
    q = q + q_dot * dt;
    
    if mod(k, 5) == 0
        show(panda, [q; 0; 0], 'Frames', 'off', 'PreservePlot', false);
        drawnow;
    end
end
