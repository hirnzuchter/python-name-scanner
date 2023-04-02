# Python Name Scanner
## Introduction
This module contains a ```NameScanner``` class that may be used
to scan text strings for important key words. This is powered 
by TensorFlow models.
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
print(get(https://youtube.com).content)
```
Imagine all of the things you can do with this!

If you have any questions or would like to collaborate, contact me at sactoa@gmail.com.
