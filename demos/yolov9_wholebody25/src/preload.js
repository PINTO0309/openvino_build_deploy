const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  ipcRenderer: {
    send: (channel, data) => ipcRenderer.send(channel, data),
    on: (channel, func) => ipcRenderer.on(channel, (event, ...args) => func(event, ...args))
  },
  detectDevices: () => ipcRenderer.invoke('detect-devices'),
  runModel: (img, width, height, device) => ipcRenderer.invoke('run-model', img, width, height, device),
  objectDetection: (image, width, height) => ipcRenderer.invoke('object-detection', image, width, height),
});