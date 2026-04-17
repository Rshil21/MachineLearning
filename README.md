
-------------------------------------setup the enviornment----------------------------------
1. Install Python:
Download and install Python from the official website.
Ensure that Python is added to your system's PATH environment variable during installation. 
2. Install the VS Code Python Extension:
Open VS Code and go to Extensions (Ctrl+Shift+X).
Search for "Python" and install the official Microsoft Python extension. 
3. Create a Virtual Environment:
Using VS code interpreter
   Using venv (recommended for most cases):
   Open a terminal within VS Code (Ctrl+Shift+' ).
   Navigate to your project's directory in the terminal.
   Create the virtual environment: python -m venv .venv.
   VS Code might suggest you select the newly created interpreter. You can choose "Yes" to proceed.
   Alternatively, you can use the Command Palette (Ctrl+Shift+P) to find and run the "Python: Create Environment" command to create the environment automatically.
   For more details : use this link https://code.visualstudio.com/docs/python/environments
Using cmd : 

    # Navigate to your project directory
    cd C:\Users\YourName\Documents\MyProject

    # Create the virtual environment
    python -m venv venv

    # Activate the virtual environment (for Windows CMD)
    myenv\Scripts\activate.bat

    # Verify by listing packages (optional)
    pip list
    

------------------------------installing all the packages------------------------------

     created one file with the name requirements.txt with all libraries
     just used this command to install the libraries in one go-----
     command pip install -r requirements.txt    

------------------------------ Running your ipynb(notebook)-----------------------------    
     create a new .ipynb file if not exists
     and click on "RUN ALL" to execute it in vs code 
