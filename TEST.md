1.  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
2. pip install -r requirements.txt  
3. python scripts\split_data.py
    Создастся папка data/ и data/dataset
    Добавить папку data/raw/ и вставить туда все txt. и png. 
4. python -m scripts.build_index
5. uvicorn app.main:app --reload