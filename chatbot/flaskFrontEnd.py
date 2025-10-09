import time
from typing import Callable

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

try:
    from .chatLib import AudioPlayer as ap
    from .chatLib import WhisperListener as wl
    from .chatLib.text_utils import clean_markdown_for_tts
    from .chatLib.util import *
    from .ifabChatWebSocket import IfabChatWebSocket
    from .welcomePage import *
except ImportError:
    from chatLib import AudioPlayer as ap
    from chatLib import WhisperListener as wl
    from chatLib.text_utils import clean_markdown_for_tts
    from chatLib.util import *
    from ifabChatWebSocket import IfabChatWebSocket
    from welcomePage import *
"""
Flask WebSocket server per la comunicazione con il bot 
@param url:                 URL del bot
@param auth:                Token di autenticazione per il bot
@param jobStation_list_top: Lista di dizionari contenenti il testo, il percorso dell'immagine per i pulsanti statici e la key del pulsante
                                [{text:"...", image_path:"...", say:"...", key:"..."}, ...]
@param machine_list_bot:    Lista di dizionari contenenti il testo, il percorso dell'immagine per i pulsanti statici e la key del pulsante
                                [{text:"...", image_path:"...", say:"...", key:"..."}, ...]
@param sttFun:              Funzione di callback per la trascrizione audio (opzionale)
                            @param sttFun(pathToAudio) -> Transcription | None
@param ttsFun:              Funzione di callback per la sintesi vocale (opzionale)
                            @param ttsFun(text) -> None
"""


def create_app(url: str, auth: str,
               jobStation_list_top: list[dict[str, str, str, str]],
               machine_list_bot: list[dict[str, str, str, str]],
               sttFun: Callable[[str], str | None] = None,
               ttsFun: Callable[[str], None] = None,
               goBotFun: Callable[[str|None], None] = None,
               getBotStatusFun: Callable[[], str] = None,
               updateBotFaceFun: Callable[[str], None] = None) -> tuple[Flask, SocketIO, IfabChatWebSocket]:
    """Crea e restituisce l'istanza dell'app Flask, socketio e client WebSocket, con tutti i callback"""

    def send_to_copilot(text: str):
        """Invia un messaggio al bot"""
        if getBotStatusFun is not None:
            botStatus = getBotStatusFun()  # Chiedi al sistema preposto lo stato del bot per inviarlo al chatbot
            text = f"Stato del bot rilevato:\n{botStatus}\n\nDomanda dell'utente:\n{text}"  # Aggiungi lo stato del bot al messaggio
        messageBox("Invio messaggio al bot", text, StyleBox.Dash_Light)
        # TODO: robot thinking face ?
        chat_client.send_message(text)

    # Callback per gestire l'inoltro dei messaggi dal backend (bot o stt) al frontend
    # se ho un messaggio ID, allora devo aggiornare quel baloon
    def backEnd_msg2UI(text, message_id=None, audio_enable=True):
        text = text.replace("Il contenuto generato dall'IA potrebbe essere errato", "")
        """Callback function for when a message is received from the bot"""
        if not message_id:  # Nessuno ID messaggio, quindi è un messaggio normale
            messageBox("Send new message to frontEnd", text, StyleBox.Dash_Light)
            message_data = {'type': 'message', 'text': text}
            if ttsFun and audio_enable:
                # Pulisci il testo da elementi Markdown prima di inviarlo al TTS
                clean_text = clean_markdown_for_tts(text)
                messageBox("Send to TTS", clean_text, StyleBox.Dash_Light)
                ttsFun(clean_text)
            # TODO: Messaggio ricevuto, fine della faccietta pensante
            socketio.emit('message', message_data)
        else:  #
            messageBox("Send to frontEnd audio transcription to append", text, StyleBox.Dash_Light)
            message_data = {'type': 'message', 'text': text, 'messageId': message_id}
            socketio.emit('stt', message_data)
        socketio.sleep(0)  # Assicurati che il messaggio venga inviato immediatamente

    def bot_err2UI(error_text):
        """Callback function for when an error occurs"""
        messageBox("Errore invio al frontend", error_text, StyleBox.Error)
        socketio.emit('message', {'type': 'error', 'text': error_text})

    # Mock function for STT (Speech-to-Text) processing
    def stt_mock(audio_path=None) -> str | None:
        """Elaborate the audio message with speech-to-text processing"""
        # This would typically involve sending the audio file to a speech-to-text service
        # and then sending the resulting text to the bot
        # For now, we'll just send a placeholder message
        if not audio_path:
            print("No audio data provided")
            return None
        print("Audio data received")
        time.sleep(1)  # Simulate processing time
        return f"Trascrizione del messaggio, Mock per {os.path.basename(audio_path)}"

    # Inizializzo gli oggetti e li configuro per l'interfaccia grafica
    chat_client = IfabChatWebSocket(url, auth)  # Inizializza il client WebSocket verso il bot
    app = Flask(__name__, static_folder='web-client')  # Creo l'istanza dell'app Flask e imposto la cartella statica
    CORS(app)  # Abilita CORS per tutte le route
    socketio = SocketIO(app, cors_allowed_origins="*")  # Inizializza SocketIO con CORS abilitato tra il backend python ed il frontend Flask

    # Configurazione delle callback esterne
    stt_fun = sttFun if sttFun else stt_mock  # Se non viene fornita una funzione STT, usa la funzione di mock

    # Registra i callback per gestire l'inoltro dei messaggi dal bot al frontend
    chat_client.add_message_callback(backEnd_msg2UI)
    chat_client.add_error_callback(bot_err2UI)

    # Crea una directory temporanea vuota all'avvio del server
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)  # Rimuovi la directory temporanea esistente
    os.makedirs(temp_dir, exist_ok=True)

    # Gestione dell'evento di connessione Socket.IO
    @socketio.on('connect')
    def handle_connect():
        """Gestisce l'evento di connessione di un client Socket.IO"""
        if goBotFun:
            goBotFun(None)  # Invia un nuovo target al robot
        # messageBox("Nuova connessione frontend", "Avvio nuova conversazione con il bot", StyleBox.Dash_Bold)
        # # Invia un messaggio di benvenuto all'utente
        # backEnd_msg2UI('Benvenuto! Puoi scrivere un messaggio o registrare un messaggio vocale.', audio_enable=False)
        # # Gestione più robusta della connessione
        # try:
        #     if chat_client.running:
        #         messageBox("Chiusura conversazione", "Chiudo la conversazione precedente con il bot", StyleBox.Light)
        #         chat_client.stop_conversation()
        #         time.sleep(0.5)  # Breve pausa per assicurarsi che la connessione precedente sia completamente chiusa
        #     # Tenta di avviare una nuova conversazione
        #     if not chat_client.start_conversation():
        #         messageBox("Errore connessione", "Impossibile avviare la conversazione con il bot", StyleBox.Error)
        #         # Invia un messaggio di errore al frontend
        #         socketio.emit('message', {'type': 'error', 'text': 'Impossibile avviare la conversazione con il bot'})
        # except Exception as e:
        #     messageBox("Errore connessione", f"Errore durante l'avvio della conversazione: {str(e)}", StyleBox.Error)
        #     socketio.emit('message', {'type': 'error', 'text': f'Errore durante la connessione: {str(e)}'})

    # Aggiungi una route per servire le immagini statiche
    @app.route('/images/<path:filename>')
    def serve_image(filename):
        return send_from_directory('web-client/images', filename)

    # Aggiungi una route per servire il file CSS
    @app.route('/css/<path:filename>')
    def serve_css(filename):
        return send_from_directory('web-client/css', filename)

    # Aggiungi una route per servire il file JavaScript
    @app.route('/js/<path:filename>')
    def serve_js(filename):
        return send_from_directory('web-client/js', filename)

    # Aggiungi route per servire le librerie JavaScript locali
    @app.route('/libs/<path:filename>')
    def serve_libs(filename):
        return send_from_directory('web-client/libs', filename)

    @app.route('/favicon.ico')
    def serve_favicon():
        return send_from_directory('web-client', 'favicon.ico')

    # Aggiungi una route per servire il file HTML e generare la pagina
    @app.route('/')
    def index():
        """Serve the main HTML page"""

        def mkHTMLbutton(buttons):
            html = ''
            for item in buttons:
                if "img_path" not in item or "text" not in item or "say" not in item or "key" not in item:
                    print(f"[WARNING] 'img_path' or 'text' or 'say' or 'key' not in item: {item}")
                    continue
                text = item["text"]
                img_path = item["img_path"]
                say = item["say"]
                key = item["key"]
                if os.path.exists(os.path.join(os.path.dirname(__file__), img_path)):  # Se l'immagine esiste, impostala come sfondo
                    if not img_path.startswith('/'):  # Assicurati che il percorso dell'immagine inizi con '/'
                        img_path = '/' + img_path
                    bg_img = f'style="background-image: url(\'{img_path}\')"'
                else:
                    bg_img = ''
                html += f'<button class="static-btn" data-say="{say}" data-key="{key}" {bg_img}><span>{text}</span></button>\n'
            return html

        # Crea HTML per i pulsanti di sinistra
        # Leggi il contenuto del file HTML
        with open(os.path.join(os.path.dirname(__file__), 'web-client/index.html'), 'r') as file:
            html_content = file.read()

        # Sostituisci i placeholder nel template
        html_content = html_content.replace('<!-- STATIC_BUTTONS_JOB_STATION -->', mkHTMLbutton(jobStation_list_top))
        html_content = html_content.replace('<!-- STATIC_BUTTONS_MACHINE -->', mkHTMLbutton(machine_list_bot))

        return html_content

    # Aggiungi una route per gestire la richiesta di invio del messaggio testuale
    @app.route('/robot-face-update', methods=['POST'])
    def robot_face_update():
        """Handle text message submission"""
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': 'No text provided'}), 400

        text = data['text']
        if updateBotFaceFun:
            updateBotFaceFun(text)

        return jsonify({'success': True})

    # Aggiungi una route per gestire la richiesta di invio del messaggio testuale
    @app.route('/send-message', methods=['POST'])
    def send_message():
        """Handle text message submission"""
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': 'No text provided'}), 400

        text = data['text']

        # Gestione più robusta della connessione
        try:
            # Se la connessione non è attiva, tenta di riavviarla
            if not chat_client.running:
                messageBox("Riconnessione", "Tentativo di riavvio della conversazione", StyleBox.Light)
                if not chat_client.start_conversation():
                    messageBox("Errore connessione", "Impossibile avviare la conversazione", StyleBox.Error)
                    return jsonify({'success': False, 'error': 'Impossibile avviare la conversazione'}), 500
                # Breve pausa per assicurarsi che la connessione sia stabilita
                time.sleep(0.5)
            # Invia il messaggio al bot in un thread separato per non bloccare la risposta HTTP
            threading.Thread(target=send_to_copilot, args=(text,)).start()
            return jsonify({'success': True})
        except Exception as e:
            messageBox("Errore invio", f"Errore durante l'invio del messaggio: {str(e)}", StyleBox.Error)
            return jsonify({'success': False, 'error': f'Errore durante l\'invio: {str(e)}'}), 500

    # Aggiungi una route per gestire l'invio di un comando a schermo
    @app.route('/button-click', methods=['POST'])
    def button_click():
        """Handle button click events without sending to chatbot"""
        data = request.json
        if not data or 'key' not in data or 'say' not in data:
            return jsonify({'success': False, 'error': 'No "key" or "say" provided'}), 400

        key = data['key']
        say = data['say']

        # Stampa il testo del pulsante nel server per debug
        messageBox("Frontend al click di un pulsante invia chiave", f"key: {key}\nsay: {say}", StyleBox.Light)
        if goBotFun:
            goBotFun(key)  # Invia il nuovo target al robot
        if ttsFun:
            ttsFun(say)  # Invia il messaggio al TTS
        return jsonify({'success': True, 'key': key})

    # Aggiungi una route per gestire l'invio di un messaggio audio
    @app.route('/upload-audio', methods=['POST'])
    def send_audio():
        """Handle audio message submission"""
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400
        audio_file = request.files['audio']

        # Genera un nome file univoco con timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        temp_path = os.path.join(temp_dir, f'audio_{timestamp}.wav')

        # Salva il file audio e logga il percorso
        audio_file.save(temp_path)
        messageBox("Frontend audio", f"File audio temporaneo salvato in: {temp_path}", StyleBox.Light)

        # Crea un URL relativo per il file audio
        audio_url = f"/temp/audio_{timestamp}.wav"

        # Gestione più robusta della connessione
        try:
            # Se la connessione non è attiva, tenta di riavviarla
            if not chat_client.running:
                messageBox("Riconnessione", "Tentativo di riavvio della conversazione per audio", StyleBox.Light)
                if not chat_client.start_conversation():
                    messageBox("Errore connessione", "Impossibile avviare la conversazione per audio", StyleBox.Error)
                    # Invia un messaggio di errore al frontend
                    socketio.emit('message', {'type': 'error', 'text': 'Impossibile avviare la conversazione. Riprova più tardi.'})
                    return jsonify({'success': False, 'error': 'Impossibile avviare la conversazione'}), 500
                # Breve pausa per assicurarsi che la connessione sia stabilita
                time.sleep(0.5)

            # Crea un ID messaggio basato sul timestamp
            message_id = f"audio_{timestamp}"

            # Invia il messaggio audio al bot in un thread separato per non bloccare la risposta HTTP
            def send_audio_thread(audio_path, message_id):
                stt_audio_text = stt_fun(audio_path)
                if stt_audio_text:
                    messageBox("Backend audio STT", f"Trascrizione audio: {stt_audio_text}", StyleBox.Light)
                    backEnd_msg2UI(stt_audio_text, message_id=message_id)  # Invia messaggio trascritto al frontend
                    if stt_fun is not stt_mock:  # Invia messaggio trascritto al bot solo se veramente trascritto
                        # Invia il messaggio al bot in un thread separato per non bloccare la risposta HTTP
                        threading.Thread(target=send_to_copilot, args=(stt_audio_text,)).start()
                    else:
                        time.sleep(1)  # Simula un breve ritardo per il mock
                        messageBox("Backend audio STT to Bot", "Trascrizione audio non inviata al bot, Mock STT", StyleBox.Light)
                        backEnd_msg2UI("Trascrizione audio non inviata al bot, Mock STT")  # Invia messaggio mock al frontend
                else:
                    messageBox("Backend audio STT", "Errore durante la trascrizione audio, impossibile distinguere parole", StyleBox.Light)
                    backEnd_msg2UI("Impossibile trascrivere il messaggio audio, troppo corto o rumoroso", message_id=message_id)
                    time.sleep(1)  # Breve attesa per non bloccare la macchina a stati
                    backEnd_msg2UI("Mi spiace ma non ho capito nulla, puoi ripetere da capo?")  # Invia messaggio frontend per ripetere

            threading.Thread(target=send_audio_thread, args=(temp_path, message_id,)).start()
            return jsonify({'success': True, 'file_path': audio_url, 'message_id': message_id})

        except Exception as e:
            messageBox("Errore audio", f"Errore durante l'elaborazione dell'audio: {str(e)}", StyleBox.Error)
            return jsonify({'success': False, 'error': f'Errore durante l\'elaborazione: {str(e)}'}), 500

    # Aggiungi una route per servire i file audio temporanei
    @app.route('/temp/<path:filename>')
    def serve_audio(filename):
        """Serve temporary audio files"""
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        return send_from_directory(temp_dir, filename)

    # Aggiungi una route per verificare lo stato della connessione
    @app.route('/check-connection', methods=['GET'])
    def check_connection():
        """Check if the connection to the bot is active"""
        is_connected = chat_client.running
        if not is_connected:
            # Tenta di riavviare la connessione se non è attiva
            is_connected = chat_client.start_conversation()

        return jsonify({
            'connected': is_connected,
            'status': 'active' if is_connected else 'disconnected'
        })

    # Aggiungi route per check-server-status che risponde che il server è pronto
    @app.route('/check-server-status')
    def server_status():
        return jsonify({'ready': True})

    # Aggiungi una route per la pagina "Chi siamo"
    @app.route('/about')
    def about():
        """Serve the about page"""
        with open(os.path.join(os.path.dirname(__file__), 'web-client/about.html'), 'r') as file:
            html_content = file.read()
        return html_content

    # Gestione dell'evento di disconnessione Socket.IO
    @socketio.on('disconnect')
    def handle_disconnect():
        """Gestisce l'evento di disconnessione di un client Socket.IO"""
        messageBox("Disconnessione frontend", "Client disconnesso", StyleBox.Light)
        # Non chiudiamo la conversazione qui, poiché potrebbe essere un refresh della pagina
        # e vogliamo mantenere la conversazione attiva per quando l'utente si riconnette

    return app, socketio, chat_client


""" Utility function for Argvparser"""


def flaskFrontEnd_argsAdd(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    flaskFrontEndParser = parser.add_argument_group("Flask WebSocket server")
    flaskFrontEndParser.add_argument('--host', type=str, default='0.0.0.0', help="Host del server [default '%(default)s']")
    flaskFrontEndParser.add_argument('--port', type=int, default=8000, help="Porta del server [default '%(default)s']")
    return flaskFrontEndParser


def flaskFrontEnd_useArgs(args: argparse.Namespace) -> tuple[str, int]:
    return args.host, args.port


if __name__ == '__main__':
    def newSetPointMock(key):
        print(f"Nuovo setpoint per il robot: '{key}'")


    parser = TreeParser(formatter_class=formatHelp, description='Avvia il server Flask con SocketIO')
    flaskFrontEnd_argsAdd(parser)  # Aggiungi gli argomenti per il server Flask
    ap.audioPlayer_argsAdd(parser)  # Aggiungi gli argomenti per l'AudioPlayer
    wl.whisperListener_argsAdd(parser)  # Aggiungi gli argomenti per il WhisperListener
    args = parser.parse_args()

    # Ottieni gli argomenti per il server Flask
    host, port = flaskFrontEnd_useArgs(args)

    # Avvia il server temporaneo di welcome-page
    run_temp_server(port)

    # Inizializza TTS
    player = ap.audioPlayer_useArgs(args)

    # Inizializza STT
    listener, listener_ready_event = wl.whisperListener_useArgs(args)


    # Inizializza il client WebSocket per la comunicazione con il bot
    # Token Bot Ema:

    # Lista di pulsanti statici (testo, percorso_immagine, testo da dire, chiave del dizionario da cui è stato generato)
    zone_lavoro = [
        {"text": "Zona saldatura", "img_path": "web-client/images/help.jpeg", "say": "Vado a saldare", "key": "saldatura"},
        {"text": "Zona debug", "img_path": "web-client/images/weather.jpg", "say": "Mi dirigo alla strumentazione di analisi", "key": "debug"},
        {"text": "Zona prototipazione", "img_path": "images/news.jpg", "say": "Vado sul tavolo di prototipazione", "key": "prototipazione"}
    ]
    macchinari = [
        {"text": "Tagliatrice Laser", "img_path": "web-client/images/info.jpg", "say": "Vado dalla Tagliatrice Laser", "key": "laser"},
        {"text": "Stampante 3D", "img_path": "web-client/images/commands.jpg", "say": "Vado verso la Stampante 3D", "key": "3d"},
        {"text": "CNC", "img_path": "web-client/images/music.jpg", "say": "Mi dirigo verso la CNC", "key": "cnc"},
        {"text": "Plotter", "img_path": "web-client/images/info.jpg", "say": "Sto andando dal Plotter", "key": "plotter"}
    ]
    # Crea l'app Flask e SocketIO con tutte le callback e le informazioni del progetto
    app, socketio, chat_client = create_app(url, auth, zone_lavoro, macchinari,
                                            ttsFun=player.play_text, sttFun=listener,
                                            goBotFun=newSetPointMock)
    
    wl.wait_for_model_loading(listener_ready_event)

    # Ferma il server temporaneo prima di avviare quello Flask
    stop_temp_server()

    # Avvia il server Flask con SocketIO
    socketio.run(app, host=host, port=port, debug=True, allow_unsafe_werkzeug=True, use_reloader=False)  # Avvia il server Flask con SocketIO disabilitando il riavvio automatico
