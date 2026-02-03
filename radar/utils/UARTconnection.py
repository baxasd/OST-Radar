import serial
from serial.tools import list_ports
import platform
import time
import sys

# Logger
import logging
log = logging.getLogger(__name__)
from radar.utils.parseFrame import *


def readAndParseUartDoubleCOMPort(dataCom):
    UART_MAGIC_WORD = bytearray(b'\x02\x01\x04\x03\x06\x05\x08\x07')
    # Find magic word, and therefore the start of the frame
    index = 0
    magicByte = dataCom.read(1)
    frameData = bytearray(b'')
    while (1):
        # If the device doesn't transmit any data, the COMPort read function will eventually timeout
        # Which means magicByte will hold no data, and the call to magicByte[0] will produce an error
        # This check ensures we can give a meaningful error
        if (len(magicByte) < 1):
            log.error("ERROR: No data detected on COM Port, read timed out")
            log.error("\tBe sure that the device is in the proper mode, and that the cfg you are sending is valid")
            magicByte = dataCom.read(1)
            
        # Found matching byte
        elif (magicByte[0] == UART_MAGIC_WORD[index]):
            index += 1
            frameData.append(magicByte[0])
            if (index == 8): # Found the full magic word
                break
            magicByte = dataCom.read(1)
            
        else:
            # When you fail, you need to compare your byte against that byte (ie the 4th) AS WELL AS compare it to the first byte of sequence
            # Therefore, we should only read a new byte if we are sure the current byte does not match the 1st byte of the magic word sequence
            if (index == 0): 
                magicByte = dataCom.read(1)
            index = 0 # Reset index
            frameData = bytearray(b'') # Reset current frame data
    
    # Read in version from the header
    versionBytes = dataCom.read(4)
    
    frameData += bytearray(versionBytes)

    # Read in length from header
    lengthBytes = dataCom.read(4)
    frameData += bytearray(lengthBytes)
    frameLength = int.from_bytes(lengthBytes, byteorder='little')
    
    # Subtract bytes that have already been read, IE magic word, version, and length
    # This ensures that we only read the part of the frame in that we are lacking
    frameLength -= 16 

    # Read in rest of the frame
    frameData += bytearray(dataCom.read(frameLength))

    # frameData now contains an entire frame, send it to parser
    outputDict = parseStandardFrame(frameData)

    return outputDict

def sendCfg(fname, cliCom):

    with open(fname, "r") as cfg_file:
        cfg = cfg_file.readlines()

    # Remove empty lines from the cfg
    cfg = [line for line in cfg if line != '\n']
    # Ensure \n at end of each line
    cfg = [line + '\n' if not line.endswith('\n') else line for line in cfg]
    # Remove commented lines
    cfg = [line for line in cfg if line[0] != '%']

    for line in cfg:
        time.sleep(.03) # Line delay

        if(cliCom.baudrate == 1250000):
            for char in [*line]:
                time.sleep(.001) # Character delay. Required for demos which are 1250000 baud by default else characters are skipped
                cliCom.write(char.encode())
        else:
            cliCom.write(line.encode())
            
        ack = cliCom.readline()
        print(ack, flush=True)
        ack = cliCom.readline()
        print(ack, flush=True)

        splitLine = line.split()
        if(splitLine[0] == "baudRate"): # The baudrate CLI line changes the CLI baud rate on the next cfg line to enable greater data streaming off the xWRL device.
            try:
                cliCom.baudrate = int(splitLine[1])
            except:
                log.error("Error - Invalid baud rate")
                sys.exit(1)
    # Give a short amount of time for the buffer to clear
    time.sleep(0.03)
    cliCom.reset_input_buffer()
    # NOTE - Do NOT close the CLI port because 6432 will use it after configuration


def sendLine(cliCom, line):

    cliCom.write(line.encode())
    ack = cliCom.readline()
    print(ack)
    ack = cliCom.readline()
    print(ack)


def find_TI_ports():
    # Find all Com Ports
    serialPorts = list(list_ports.comports())

    
    if platform.system().count("Windows"):
        CLI_XDS_SERIAL_PORT_NAME = 'XDS110 Class Application/User UART'
        DATA_XDS_SERIAL_PORT_NAME = 'XDS110 Class Auxiliary Data Port'
        CLI_SIL_SERIAL_PORT_NAME = 'Enhanced COM Port'
        DATA_SIL_SERIAL_PORT_NAME = 'Standard COM Port'

        # Find default CLI Port and Data
        p = {}
        for port in serialPorts:
            if (
                CLI_XDS_SERIAL_PORT_NAME in port.description
                or CLI_SIL_SERIAL_PORT_NAME in port.description
            ):
                p = {"UART": port}
                print(f"CLI COM Port found: {port.device}")

            elif (
                DATA_XDS_SERIAL_PORT_NAME in port.description
                or DATA_SIL_SERIAL_PORT_NAME in port.description
            ):
                print(f"Data COM Port found: {port.device}")
                p = {"DATA": port}

    elif platform.system().count("Darwin"):
        CLI_XDS_SERIAL_PORT_NAME = 'usbmodemR00810381'
        DATA_XDS_SERIAL_PORT_NAME = 'usbmodemR00810384'
            # Find default CLI Port and Data
        p = {}
        for port in serialPorts:
            if (
                CLI_XDS_SERIAL_PORT_NAME in port.name
            ):
                print(f"CLI COM Port found: {port.device}")
                p["UART"] = port

            elif (
                DATA_XDS_SERIAL_PORT_NAME in port.name
            ):
                print(f"Data COM Port found: {port.device}")
                p["DATA"] = port
    return p


def connectComPorts(cliCom, dataCom):
    # Longer timeout time for xWRL6432 to support applications with low power / low update rate
    cliCom = serial.Serial(cliCom, 115200, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0.6)
    dataCom = serial.Serial(dataCom, 921600, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0.6)
    dataCom.reset_output_buffer()
    return cliCom, dataCom