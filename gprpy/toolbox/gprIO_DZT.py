import struct
import numpy as np


def readdzt(filename, start_scan=0, n_scans=-1):
    '''
    Reads a GSSI .DZT data file.

    Args:
        filename (str): Data file name including .DZT extension.
        start_scan (int): First scan to read, by index. Default is 0 (read from
            start of data.)
        n_scans (int): Number of scan traces to read. Default is -1 (read all).

    Returns:
        ndarray: data matrix whose columns contain the traces.
        dict: dict with information from the header.

    Thanks to Ian Nesbitt for pointing out extended headers and
    providing the documentation file. Documentation file is
    DZT.File.Format.6-14-16.pdf
    '''

    # H is unsigned int 16 (ushort = uint16)
    # h is short (int16)
    # I is unsigned int 32 (uint = uint32)
    # i is int32
    # f is float

    minheadsize = 1024

    info = {}

    with open(filename, 'rb') as fid:
        info["rh_tag"] = struct.unpack('h', fid.read(2))[0]  # Pos 00

        # Size of the header
        info["rh_data"] = struct.unpack('h', fid.read(2))[0]  # Pos 02

        # Samples per trace
        info["rh_nsamp"] = struct.unpack('h', fid.read(2))[0]  # Pos 04

        # Bits per word
        info["rh_bits"] = struct.unpack('h', fid.read(2))[0]  # Pos 06

        # Binary offset
        info["rh_zero"] = struct.unpack('h', fid.read(2))[0]  # Pos 08

        # Scans per second
        info["rhf_sps"] = struct.unpack('f', fid.read(4))[0]  # Pos 10

        # Scans per meter
        info["rhf_spm"] = struct.unpack('f', fid.read(4))[0]  # Pos 14

        # Meters per mark
        info["rhf_mpm"] = struct.unpack('f', fid.read(4))[0]  # Pos 18

        # Startposition [ns]
        info["rhf_position"] = struct.unpack('f', fid.read(4))[0]  # Pos 22

        # length of trace [ns]
        info["rhf_range"] = struct.unpack('f', fid.read(4))[0]  # Pos 26

        # Number of passes
        info["rh_npass"] = struct.unpack('h', fid.read(2))[0]  # Pos 30

        # Creation date and time
        info["rhb_cdt"] = struct.unpack('f', fid.read(4))[0]  # Pos 32

        # Last modified date & time
        info["rhb_mdt"] = struct.unpack('f', fid.read(4))[0]  # Pos 36

        # no idea
        info["rh_mapOffset"] = struct.unpack('h', fid.read(2))[0]  # Pos 40

        # No idea
        info["rh_mapSize"] = struct.unpack('h', fid.read(2))[0]  # Pos 42

        # offset to text
        info["rh_text"] = struct.unpack('h', fid.read(2))[0]  # Pos 44

        # Size of text
        info["rh_ntext"] = struct.unpack('h',  fid.read(2))[0]  # Pos 46

        # offset to processing history
        info["rh_proc"] = struct.unpack('h', fid.read(2))[0]  # Pos 48

        # size of processing history
        info["rh_nproc"] = struct.unpack('h', fid.read(2))[0]  # Pos 50

        # number of channels
        info["rh_nchan"] = struct.unpack('h', fid.read(2))[0]  # Pos 52

        # ... and more stuff we don't really need.

    # Define data type based on words per bit.
    # From documentation: Eight byte and sixteen byte samples are
    # unsigned integers. Thirty-two bit samples are signed integers.
    if info["rh_bits"] == 8:
        datatype = np.uint8  # unsigned int
    elif info["rh_bits"] == 16:
        datatype = np.uint16  # unsigned int
    elif info["rh_bits"] == 32:
        datatype = np.int32

    # Offset will tell us how long the header is in total.
    # There could be a minimal header of 1024 bits, or
    # very old files may have had 512 bits.
    if info["rh_data"] < minheadsize:
        head_offset = minheadsize * info["rh_data"]
    else:
        head_offset = minheadsize * info["rh_nchan"]

    # Deal with offset
    try:
        start_offset = int(start_scan * info['rh_nchan'] * info['rh_nsamp'] * info['rh_bits'] / 8)
    except:
        warnings.warn("Unable to set start_offset from start_scan.")
        start_offset = 0
    if n_scans > 0:
        try:
            count = int(n_scans * info['rh_nsamp'] * info['rh_nchan'])
        except:
            warnings.warn("Unable to set count from n_scans.")
            count = -1
    else:
        count = -1

    # Read the file
    datvec = np.fromfile(filename,
                         dtype=datatype,
                         offset=head_offset + start_offset,
                         count=count,
                         )

    # Turn unsigned integers into signed integers
    # Only necessary where unsigned
    if info["rh_bits"] == 8 or info["rh_bits"] == 16:
        datvec = datvec.astype(np.int32) - 2**info["rh_bits"] // 2

    # Reshape into matrix
    data = datvec.reshape(-1, info["rh_nsamp"]*info["rh_nchan"]).T

    return data, info
