# Image search engine
Search engine for image data

# Enviroment
Setup
```
pip install -r enviroment/requirements.txt
```
# Script
Run local for test api
```
uvicorn main:app --reload
```
# Calculate mAP use evaluate
In windows use Visual C++ 2015 x86 Native Tools Command Prompt to build exe from compute_ap.cpp.
Build compute_ap.exe
```
cl /EHsc compute_ap.cpp
```
Run evaluate:
```
python evaludate.py
```
# Reference
