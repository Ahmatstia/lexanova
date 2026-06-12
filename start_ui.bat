@echo off
echo ====================================================
echo    Memulai Lexa Assistant Frontend (UI)
echo ====================================================
echo.
echo Pastikan backend uvicorn dan Ollama sudah berjalan!
echo.
echo Membuka browser...

cd frontend
start http://localhost:3000
python -m http.server 3000
