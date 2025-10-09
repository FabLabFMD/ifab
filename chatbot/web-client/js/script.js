document.addEventListener('DOMContentLoaded', function () {
    

    // Aggiungi event listener per i pulsanti statici
    document.querySelectorAll('.static-btn').forEach(button => {
        button.addEventListener('click', function () {
            const say = this.dataset.say;
            const key = this.dataset.key;

            // Invia il testo del pulsante al server
            // showLoading(`${say}`);             // Mostra l'animazione di caricamento temporanea per i pulsanti
            fetch('/button-click', {
                method: 'POST', headers: {
                    'Content-Type': 'application/json'
                }, body: JSON.stringify({key: key, say: say})
            })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.success) {
                        // Mostra un feedback visivo che il pulsante è stato premuto
                        this.classList.add('button-pressed');
                        setTimeout(() => {
                            this.classList.remove('button-pressed');
                        }, 300);

                        console.log('Key inviata:', key);
                        console.log('Messaggio inviato:', say);
                        // Imposta un timer per nascondere automaticamente il loading dopo 1 secondi dalla conferma di ricezione
                        setTimeout(hideLoading, 1000); // 1 secondi
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    addErrorMessage("Si è verificato un errore durante l'invio del comando.", isGui = true);
                });
        });
    });
});