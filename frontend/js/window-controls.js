// Управление окном Electron
const { ipcRenderer } = require('electron');

document.addEventListener('DOMContentLoaded', function() {
    const closeBtn = document.getElementById('closeBtn');
    const minimizeBtn = document.getElementById('minimizeBtn');
    const maximizeBtn = document.getElementById('maximizeBtn');
    
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            window.close();
        });
    }
    
    if (minimizeBtn) {
        minimizeBtn.addEventListener('click', () => {
            // Отправляем сообщение в main процесс для минимизации
            if (window.electronAPI) {
                window.electronAPI.minimize();
            } else {
                // Альтернативный способ через ipcRenderer
                ipcRenderer.send('minimize-window');
            }
        });
    }
    
    if (maximizeBtn) {
        maximizeBtn.addEventListener('click', () => {
            // Отправляем сообщение в main процесс для максимизации
            if (window.electronAPI) {
                window.electronAPI.maximize();
            } else {
                ipcRenderer.send('maximize-window');
            }
        });
    }
});
