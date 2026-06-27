// PM2 config for XLiteOCR. Deploy with: pm2 start ecosystem.config.js
// Reload (never restart) on subsequent deploys: pm2 reload xlite-ocr
module.exports = {
  apps: [
    {
      name: "xlite-ocr",
      // cwd defaults to the directory you run `pm2 start` from. Set it
      // explicitly if you launch pm2 from elsewhere.
      script: "venv/bin/uvicorn",
      args: "app.server:app --host 127.0.0.1 --port 3011 --workers 1",
      interpreter: "none",
      autorestart: true,
      max_restarts: 10,
      env: {
        // Do NOT set OMP_NUM_THREADS — PaddleOCR cpu_threads governs threading
        // and OMP_NUM_THREADS triggers an OpenBLAS warning / can hurt throughput.
        PYTHONUNBUFFERED: "1",
        XLITE_FORMULA: "0", // formula (LaTeX) model off by default — keep it light
      },
    },
  ],
};
