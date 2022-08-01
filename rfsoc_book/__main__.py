# library
import sys
import os
import subprocess
import shutil
from distutils.dir_util import copy_tree

# variables
package_name = 'rfsoc_book'
board_list = ['ZCU111', 'RFSoC2x2', 'RFSoC4x2']

# dialogue
help_dialogue = ''.join(['\r\nThe rfsoc_book module accepts one of the following arguments:', '\r\n',
                         '* install : Installs the notebooks and packages to the Jupyter directory', '\r\n',
                         '* uninstall : Uninstalls the notebooks and packages from the Jupyter directory', '\r\n',
                         '* clean : Returns the notebooks to their original states', '\r\n',
                         '* help : Displays this dialogue', '\r\n'])

error_dialogue = '\r\nUnknown error occurred.\r\n'

# check arguments
args = sys.argv
if len(args) > 2:
    raise RuntimeError(help_dialogue)
    
arg = args[1]
if arg not in ['install', 'uninstall', 'clean', 'help', 'unpackage']:
    raise ValueError(help_dialogue)
    
# define functions
def install_notebooks():
    print('\r\n***** Installing Notebooks *****\r\n')
    dst = get_notebook_dst()
    src = os.path.abspath(os.path.join(os.path.realpath(__file__), '..', 'notebooks'))
    if os.path.exists(dst):
        raise RuntimeError(''.join(['Notebooks already installed. ',
                                    'Please uninstall notebooks before reinstalling.\r\n']))
    if not os.path.exists(src):
        raise RuntimeError(''.join(['Path does not exist: ', src, '\r\n']))
    copy_tree(src, dst)
    logfile = os.path.abspath(os.path.join(os.path.realpath(__file__), '..', 'install.txt'))
    with open(logfile, 'w') as f:
        f.write(dst)
        
def install_packages():
    dst = os.path.abspath(os.path.join(os.path.realpath(__file__), '..', 'package_list.txt'))
    if not os.path.exists(dst):
        raise RuntimeError(error_dialogue)
    with open(dst, 'r') as f:
        package_list = f.readlines()
    for package in package_list:
        package_name, package_src = package.split(' ')
        print(''.join(['***** Installing ', package_name, ' *****\r\n']))
        status = subprocess.check_call(["pip3", "install", package_src])
        print('\r\n') # Pip is not playing nice
        if status != 0:
            raise RuntimeError(''.join(['Package ', package_src, ' failed to install.\r\n']))

def uninstall_notebooks():
    print('\r\n***** Uninstalling Notebooks *****\r\n')
    logfile = os.path.abspath(os.path.join(os.path.realpath(__file__), '..', 'install.txt'))
    if not os.path.exists(logfile):
        raise RuntimeError('Notebooks do not have an install location. Nothing has been removed.\r\n')
    with open(logfile, 'r') as f:
        dst = f.readline()
    if not os.path.exists(dst):
        raise RuntimeError('Notebooks are not installed. Nothing has been removed.\r\n')
    shutil.rmtree(dst)
    os.remove(logfile)
    print('Notebooks uninstalled successfully.\r\n')
    
def uninstall_packages():
    dst = os.path.abspath(os.path.join(os.path.realpath(__file__), '..', 'package_list.txt'))
    if not os.path.exists(dst):
        raise RuntimeError(error_dialogue)
    with open(dst, 'r') as f:
        package_list = f.readlines()
    for package in package_list:
        package_name, _ = package.split(' ')
        print(''.join(['***** Uninstalling ', package_name, ' *****\r\n']))
        status = subprocess.check_call(["pip3", "uninstall", "-y", package_name])
        print('\r\n') # Pip is not playing nice
        if status != 0:
            raise RuntimeError(''.join(['Package ', package_name, ' failed to uninstall.\r\n']))
            
def unpackage_notebooks():
    print('\r\n***** Unpackaging Notebooks *****\r\n')
    if 'BOARD' not in os.environ:
        board = 'RFSoC4x2'
    else:
        board = os.environ['BOARD']
        if board not in board_list:
            board = 'RFSoC4x2'
    currentdir = get_notebook_dst()
    for folder in os.listdir(currentdir):
        notebookdir = os.path.join(currentdir, folder, 'boards')
        if os.path.exists(notebookdir):
            for file in os.listdir(notebookdir):
                file_split = file.split('_')
                if board in file_split:
                    file_split.remove(board)
                    file_name = '_'.join(file_split)
                    src = os.path.join(notebookdir, file)
                    dst = os.path.join(notebookdir, '..', file_name)
                    shutil.copy(src, dst)

def clean_notebooks():
    print('\r\n***** Cleaning Notebooks *****\r\n')
    uninstall_notebooks()
    install_notebooks()
    unpackage_notebooks()
    print('Notebooks cleaned successfully\r\n')

def get_notebook_dst():
    if 'PYNQ_JUPYTER_NOTEBOOKS' not in os.environ:
        # Install to current working directory if not on PYNQ
        dst = os.path.join(os.getcwd(), package_name)
        dialogue = ''.join(['Not using a PYNQ board. ', 
                            'Use `export PYNQ_JUPYTER_NOTEBOOKS=<desired-notebook-path>` ',
                            'to set the notebooks directory.\r\n',
                            'Installing notebooks to the current working directory: ',
                            dst, '\r\n'])
    else:
        dst = os.path.join(os.environ['PYNQ_JUPYTER_NOTEBOOKS'], package_name)
        dialogue = ''.join(['Using a PYNQ board. ',
                            'Installing notebooks to the PYNQ Jupyter directory: ',
                            dst, '\r\n'])
    print(dialogue)
    return dst

# run script
if arg == 'install':
    install_notebooks()
    unpackage_notebooks()
    if 'BOARD' not in os.environ:
        print('Not installing RFSoC packages.\r\n')
    else:
        board = os.environ['BOARD']
        if board in board_list:
            print('Installing RFSoC packages...\r\n')
            install_packages()
        else:
            print('Not installing RFSoC packages as not an RFSoC platform.\r\n')
    print('rfsoc_book installed successfully\r\n')
elif arg == 'uninstall':
    uninstall_notebooks()
    #if 'BOARD' not in os.environ:
    #    print('Not uninstalling RFSoC packages.\r\n')
    #else:
    #    board = os.environ['BOARD']
    #    if board in board_list:
    #        print('Uninstalling RFSoC packages...\r\n')
    #        uninstall_packages()
    #    else:
    #        print('Not uninstalling RFSoC packages as not an RFSoC platform.\r\n')
    print('rfsoc_book uninstalled successfully\r\n')
elif arg == 'clean':
    clean_notebooks()
elif arg == 'help':
    print(help_dialogue)
elif arg == 'unpackage':
    unpackage_notebooks()
else:
    raise RuntimeError(error_dialogue)
    
