// Tells the page it's inside the Encore app (REC shows only here) and lets its Quit close the app.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('encoreApp', { quit: () => ipcRenderer.send('encore-quit') });
