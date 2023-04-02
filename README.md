# Python Name Scanner
## Introduction
This module contains a ```NameScanner``` class that may be used
to scan text strings for important key words. This is powered 
by TensorFlow models. To build your own keyword-finding models, 
look here: https://github.com/hirnzuchter/name-or-not. For using your
model with pyns, you need to save your tokenizer and your weights.
## Dependencies
Tensorflow is the only dependency for this tool.
## Recommended Usage
After downloading these files to your project directory, 
create a python file. Import the NameScanner and use find_names
on your text. This is very powerful with web access. Try:
```
from requests import get
from pyns import NameScanner

ns = NameScanner(quantity=20, bothgrams=True)
print(ns.find_names(str(get("https://youtube.com").content)))
```
This may be used for things like marketing or even more fine-tuned searching.


If you have any questions or would like to collaborate, contact me at sactoa@gmail.com.
