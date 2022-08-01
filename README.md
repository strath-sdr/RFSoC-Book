<img src="./rfsoc_book/notebooks/common/rfsoc_book_banner.jpg" width="100%">

# Software Defined Radio with Zynq® UltraScale+ RFSoC
This repository contains the companion Jupyter Notebooks for the Software Defined Radio with Zynq® UltraScale+ RFSoC book.

To begin installing the Jupyter Notebooks on your system, click-on one of the options below.
* [RFSoC Setup](#rfsoc-setup)
* [Notebook Installation](#notebook-installation)

## RFSoC Setup
This repository is currently only compatible with [RFSoC-PYNQ v2.7](https://github.com/Xilinx/PYNQ/releases). Follow the steps below to setup the RFSoC platform for installing the companion Jupyter Notebooks.

1. Currently, there are only 3 compatible RFSoC platforms. These are the [RFSoC4x2](http://rfsoc-pynq.io/), [ZCU111](https://www.xilinx.com/products/boards-and-kits/zcu111.html), and [RFSoC2x2](http://rfsoc-pynq.io/).

2. Install PYNQ v2.7 onto an SD card and plug it in to your RFSoC platform.

3. Your RFSoC platform requires internet access to install the companion Jupyter Notebooks. Follow the instructions [here](https://pynq.readthedocs.io/en/v3.0.0/getting_started/network_connection.html?highlight=internet) that assist with internet access.

4. Navigate to JupyterLab by opening a browser (preferably Chrome) and connecting to `http://<board_ip_address>:9090/lab`.

## Notebook Installation
The companion Jupyter Notebooks can be installed on a computer or RFSoC platform. Follow the instructions below to install the notebooks through JupyterLab. If you haven't already, launch JupyterLab on your computer or RFSoC platform.

1. We need to open a terminal in JupyterLab. Firstly, open a launcher window as given in the following figure.

<p align="center">
  <img src="../main/open_jupyter_launcher.jpg" width="35%" />
<p/>

2. Now open a terminal in Jupyter as shown below:

<p align="center">
  <img src="../main/open_terminal_window.jpg" width="35%" />
<p/>

3. Install the RFSoC Book notebooks through PIP by executing the following command in the terminal.

```sh
pip install https://github.com/strath-sdr/RFSoC-Book/archive/v1.0.0.tar.gz
```

4. Run the following command in the Jupyter terminal window to install the notebooks and dependencies.

```sh
python -m rfsoc_book install
```

5. The RFSoC-Book notebooks are installed. Navigate to the JupyterLab workspace and you will find the notebooks in a folder named `rfsoc_book`.

## Additional Commands
The RFSoC Book module provides additional commands for those that would like to clean notebooks after use, or uninstall notebooks from their system.

Notebooks can be reset by running the following command in the Jupyter terminal.

```sh
python -m rfsoc_book clean
```

To uninstall all notebooks and dependencies, run the command below in the Jupyter terminal.

```sh
python -m rfsoc_book uninstall
```

## Warning and Disclaimer
The best efforts of the authors have been used to ensure that accurate and current information is presented in this repository. This includes researching the topics covered and developing examples. The material included is provided on an "as-is" basis in the best of faith, and the authors do not make any warranty of any kind, expressed, or implied, with regard to the documentation contained in this repository. The authors shall not be held liable for any loss or damage resulting directly or indirectly from any information or examples contained herein.

---
Copyright © 2023 Strathclyde Academic Media

---
---
