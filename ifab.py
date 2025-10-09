import json
import math
import socket
import threading

from chatbot.flaskFrontEnd import *
from vision.vision import *

version = "0.0.1"


class RobotController:
    def __init__(self, client_addr: str, client_port: int, targets: dict, table: dict):
        # Configurazione client
        self.client_addr = client_addr
        self.client_port = client_port

        # Memoria per i dati più recenti
        self.memory = {
            'robot': {'data': None},
            'markers': {}
        }

        # Variabile per tenere traccia della socket
        self.sock = None

        # Target macchina
        self.target_machine = None
        self.targets = targets if targets is not None else {}

        # Dati sul campo fisico
        self.table = table if table is not None else {}

    def get_socket(self):
        """Ottiene una socket valida o ne crea una nuova."""
        if self.sock is None:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            except Exception as e:
                print(f"Errore nella creazione della socket: {e}")
        return self.sock

    def set_target(self, target: str | None):
        """Imposta una nuova macchina target."""
        self.target_machine = target
        self.send_to_robot(robot_data_fresh=False)

    def update_states(self, data: dict):
        """Aggiorna lo stato del robot e dei marker."""
        # Aggiorna la memoria del robot
        robotData = False
        if data.get('robot') is not None:
            self.memory['robot']['data'] = data['robot']
            robotData = True

        # Aggiorna la memoria dei marker
        for marker_key, marker_data in data.get('markers', {}).items():
            self.memory['markers'][marker_key] = marker_data

        # Invia i dati al robot
        self.send_to_robot(robot_data_fresh=robotData)

    def send_to_robot(self, robot_data_fresh=False):
        # Componi il pacchetto da inviare
        toSend = {}
        if robot_data_fresh and self.memory['robot']['data'] is not None:
            x = self.memory['robot']['data']["position"][0]
            y = self.memory['robot']['data']["position"][1]
            theta = self.memory['robot']['data']["angle"]
            toSend['robot'] = {"x": x, "y": y, 'theta': theta}

        if self.memory['markers'].get(self.target_machine) is not None:
            x = self.memory['markers'][self.target_machine]["position"][0]
            y = self.memory['markers'][self.target_machine]["position"][1]
            theta = self.memory['markers'][self.target_machine]["angle"]
            x_min = self.table['offset_inside']
            x_max = self.table['width'] - self.table['offset_inside']
            y_min = self.table['offset_inside']
            y_max = self.table['height'] - self.table['offset_inside']
            if x < x_min or x > x_max or y < y_min or y > y_max:
                print(f"Il target '{self.target_machine}' è fuori dai limiti dello spazio raggiungibile: {x}, {y}")
            else:
                toSend['target'] = {"x": x, "y": y, 'theta': theta}

        if self.target_machine is None and self.memory['robot']['data'] is not None:
            # Se non abbiamo un target, ma il robot è stato visto almeno una volta, gli diciamo di andare al centro del campo
            x = self.table['width'] / 2
            y = self.table['height'] / 2
            theta = math.pi / 2
            toSend['target'] = {"x": x, "y": y, 'theta': theta}

        if not toSend:  # Invia i dati usando la socket solo se c'è qualcosa da inviare
            print("Nessun dato da inviare al robot")
            return

        toSend['timestamp'] = time.time();
        #print('Data Send to robot:', toSend)
        try:
            s = self.get_socket()
            if s:
                json_data = json.dumps(toSend, indent=0).replace("\n", "")
                bytes_data = json_data.encode('utf-8') + b'\0'
                s.sendto(bytes_data, (self.client_addr, self.client_port))
        except Exception as e:
            print(f"Errore nell'invio dei dati: {e}")
            # Resetta la socket in caso di errore
            self.sock = None

    def botStatus(self):
        """Restituisce lo stato del robot in base alla sua distanza dal target."""
        status = ""
        robot_pos = self.memory['robot']['data']["position"]
        for mKey, target in self.memory['markers'].items():
            target_pos = target["position"]
            distance = math.dist(robot_pos, target_pos) * 100
            status += f"Il robot dista da '{mKey}': {distance:.1f} cm\n"

        if self.target_machine is None:
            # Verifica se abbiamo un target impostato
            status += "Attualmente non c'è nessun target impostato per il robot"
        elif (self.memory['robot']['data'] is None or self.memory['markers'].get(self.target_machine) is None):
            # Verifica se abbiamo dati del robot e del target
            status += f"Il robot si sta muovendo verso: {self.targets[self.target_machine]['text']}"
        else:
            threshold = 0.2  # Soglia di distanza in metri, 20 cm
            if distance > threshold:
                status += f"Il robot si sta dirigendo verso: {self.targets[self.target_machine]['text']}"
            else:
                status += f"Il robot si trova davanti a: {self.targets[self.target_machine]['text']}"
        return status
    
    def update_face(self, state):
        # TODO: Crea la logica con uno switch-match per inviare al robot la faccia da fare
        # idle, listen, speak
        #print(state)
        toSend = {}
        match state:
            case "listen":
                toSend["face"] = 3
            case "speak":
                toSend["face"] = 2
            case _:
                toSend["face"] = 1
        print('Data Send to robot:', toSend)
        try:
            s = self.get_socket()
            if s:
                json_data = json.dumps(toSend, indent=0).replace("\n", "")
                bytes_data = json_data.encode('utf-8') + b'\0'
                s.sendto(bytes_data, (self.client_addr, self.client_port))
        except Exception as e:
            print(f"Errore nell'invio dei dati: {e}")
            # Resetta la socket in caso di errore
            self.sock = None
        pass



def wait_for_port_available(port, host='localhost', timeout=10):
    """
    Attende che una porta si liberi entro un determinato timeout.

    Args:
        port (int): Numero della porta da verificare
        host (str): Host su cui controllare la porta
        timeout (int): Tempo massimo di attesa in secondi

    Returns:
        bool: True se la porta è disponibile, False se è ancora occupata dopo il timeout
    """
    import time
    import socket as sock

    def is_port_in_use(port, host):
        with sock.socket(sock.AF_INET, sock.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return False
            except sock.error:
                return True

    start_time = time.time()
    port_free = False

    while time.time() - start_time < timeout:
        if not is_port_in_use(port, host):
            port_free = True
            break
        print(f"La porta {port} è occupata. Attendo che si liberi...")
        time.sleep(1)

    if not port_free:
        print(f"Timeout: la porta {port} è ancora occupata dopo {timeout} secondi.")

    return port_free


if __name__ == '__main__':
    # Argomenti da riga di comando di IFAB
    parser = TreeParser(formatter_class=formatHelp, description='Avvio del sistema IFAB')
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {version}')
    parser.add_argument('-c', '--config', dest='config', help="Percorso del file di configurazione [default '%(default)s']", default='config.json')
    flaskFrontEnd_argsAdd(parser)  # Aggiungi gli argomenti per il server Flask
    ap.audioPlayer_argsAdd(parser)  # Aggiungi gli argomenti per l'AudioPlayer
    wl.whisperListener_argsAdd(parser)  # Aggiungi gli argomenti per il WhisperListener
    args = parser.parse_args()

    # Ottieni gli argomenti per il server Flask
    host, port = flaskFrontEnd_useArgs(args)

    # Avvia il server temporaneo di welcome-page
    # run_temp_server(port)

    # Inizio il caricamento in memoria di tutte le risorse dei vari sottemi
    with open(args.config) as f:
        conf = json.load(f)

    # Inizializza il client per la comunicazione con il robot
    targetMachines = merge({}, conf['workZone'], conf['macchinari'])
    robot_client = RobotController(conf['robot']['client_addr'], conf['robot']['client_port'], targets=targetMachines, table=conf['table'])
    # Avvio del sottosistema di visione
    cameraSystem = vision_setup(conf, visionStateUpdate=robot_client.update_states)
    # Inizializza TTS
    def talkFace():
        robot_client.update_face("listen")
    def endTalkFace():
        robot_client.update_face("idle")
    player = ap.audioPlayer_useArgs(args, startTalkCallback=talkFace, stopTalkCallback=endTalkFace)
    # Callback che invia la frase e attende che finisca per stoppare la faccia
    def ttsTakl_face(text):
        def sendAndWait(text):
            player.play_text(text)
            player.waitEndBuffer()
            endTalkFace()
        faceWaitThread = threading.Thread(target=sendAndWait, args=(text,))
        faceWaitThread.daemon = True  # Il thread terminerà quando il programma principale termina
        faceWaitThread.start()
        

    # Inizializza STT
    listener, whisper_ready_event = wl.whisperListener_useArgs(args)

    # Generazione dei pulsanti statici con i target (testo, percorso_immagine, testo da dire, chiave del dizionario da cui è stato generato)
    workZone = [{'key': str(key), 'text': target['text'], 'img_path': target['img_path'], 'say': target['say']} for key, target in conf['workZone'].items()]
    macchinari = [{'key': str(key), 'text': target['text'], 'img_path': target['img_path'], 'say': target['say']} for key, target in conf['macchinari'].items()]

    if wl.wait_for_model_loading(whisper_ready_event):
        print("Modello whisper caricato con successo")
    else:
        print("Timeout durante il caricamento del modello whisper")
        exit(1)

    # Ferma il server temporaneo prima di avviare quello Flask
    # stop_temp_server()

    # Crea l'app Flask e SocketIO con tutte le callback e le informazioni del progetto
    app, socketio, chat_client = create_app(conf['url'], conf['auth'], jobStation_list_top=workZone, machine_list_bot=macchinari,
                                            ttsFun=ttsTakl_face, sttFun=None,
                                            goBotFun=robot_client.set_target, getBotStatusFun=robot_client.botStatus, updateBotFaceFun=robot_client.update_face)

    # Prima di avviare il server Flask, verifica che la porta sia libera
    # if not wait_for_port_available(port, host):
    #    print("Arresto del programma a causa di porta occupata.")
    #    exit(1)

    # Avvia il server Flask con SocketIO in un thread separato

    flask_thread = threading.Thread(target=lambda: socketio.run(app, host=host, port=port, debug=True, allow_unsafe_werkzeug=True, use_reloader=False))
    flask_thread.daemon = True  # Il thread terminerà quando il programma principale termina
    flask_thread.start()
    print("Server Flask avviato in un thread separato")

    # Avvia il sistema di visione nel thread principale
    cameraSystem.run()
