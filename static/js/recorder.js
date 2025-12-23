// static/js/recorder.js
let mediaRecorder;
let recordedChunks = [];

window.recorderStart = async function() {
  recordedChunks = [];
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("getUserMedia not supported in this browser.");
    return;
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = function(e) {
    if (e.data.size > 0) recordedChunks.push(e.data);
  };
  mediaRecorder.start();
};

window.recorderStop = async function() {
  return new Promise((resolve) => {
    if (!mediaRecorder) return resolve(null);
    mediaRecorder.onstop = function() {
      const blob = new Blob(recordedChunks, { type: 'audio/webm' });
      resolve(blob);
    };
    mediaRecorder.stop();
  });
};
