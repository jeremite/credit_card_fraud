# how to run the cods
1. I created the content in __.html__ and __.ipynb__ for easy navigating to each part of the codes.
2. __.html__ is for browsing (showing the codes and outputs)
3. If you would like to run the code, you would have to put the main code file __.ipynb__, all the defined functions file __utils.py__, the list of all of the project's python dependencies file __requirements.txt__ and the feature engieering pipeline diagram __pipeline.png__ in the same directory.
4. Open the main code file __.ipynb__ notebook:
    - run the __prerequisite__ section first. This part will install (pip install) the dependencies listed in __requirements.txt__ and import them into the current jupyter notebook session.
    - Then run the __Question1,2,3__ sections in a sequence to download and explore the data. For the modelling part __Question4__, the raw data will be read again (ignoring the manipulation in the first 3 questions).

