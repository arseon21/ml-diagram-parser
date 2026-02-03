Python 3.12.8

1. py -3.12 -m venv venv     
 .\venv\Scripts\activate   
2. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
3. pip install -r requirements.txt  
4. python scripts\split_data.py
    Создастся папка data/ и data/dataset
    Добавить папку data/raw/ и вставить туда все txt. и png. 
Ещё раз запустить python scripts\split_data.py
5. python -m scripts.build_index
6. uvicorn app.main:app --reload