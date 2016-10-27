#!/usr/bin/env python

import sys
import numpy as np

def read_timetrace(filename):
  """
  Returns the time trace from the given file. Returns the time and
  voltage array, in that order.

  Both arrays are 1-D.
  """
  bwf = LecroyBinaryWaveform(filename)
  return bwf.WAVE_ARRAY_1_time, bwf.WAVE_ARRAY_1.ravel()

from contextlib import contextmanager
@contextmanager
def _open(filename, file_content):
  if file_content is None:
    fh = open(filename, 'rb')
    yield fh
    fh.close()
  else:
    try:
      from cStringIO import StringIO
    except:
      from StringIO import StringIO

    fh = StringIO(file_content)
    yield fh
    fh.close()

class LecroyBinaryWaveform(object):
  """
  Implemented according to specs at:

    http://teledynelecroy.com/doc/docview.aspx?id=5891

  Partially derived from lecroy.py from:

    http://qtwork.tudelft.nl/gitdata/users/guen/qtlabanalysis/analysis_modules/general/lecroy.py
  """

  def __init__(self, inputfilename, file_content=None, count=-1):
    """
    inputfilename: path to .trc file to read
    file_content: if given, will be used in place of data on disk. Useful when
                  loading data from zips
    count: number of samples to read (default value is -1 which means reading all samples)
    """
    super(LecroyBinaryWaveform, self).__init__()
    self._count = count
    self._inputfilename = inputfilename
    self._file_content = file_content

    self._read_header()
    self._WAVE_ARRAY_1 = None
    self._WAVE_ARRAY_RAW = None

  def _read_header(self):
    with _open(self._inputfilename, self._file_content) as fh:
      header = fh.read(50)
      self.aWAVEDESC = header.decode('ascii').find('WAVEDESC')

      def at(offset):
        return self.aWAVEDESC + offset

      # the lecroy format says COMM_ORDER is an enum, which is a 16 bit
      # value and therefore subject to endianness. However COMM_ORDER
      # dictates the endianness! However, since the possible values are
      # either 0, which is the same in either endianness, or 0x1 or 0x7000
      # in big/small endianness, we can just check for 0. Since all read_*
      # methods needs a define endianness, we can default to 0 and not
      # worry about being wrong because of the preceding argument.

      # XXX The attribute names are important! Any attribute that is all
      # caps and does not start with '_' is considered metadata and will
      # be exported as part of the metadata property. This means it will
      # also be written to file when saving as CSV

      self.COMM_ORDER = 0

      # We do a double read because after the first read, we will know the
      # correct endianness based on the above argument, and therefore will
      # have the correct value for COMM_ORDER. Otherwise 1 becomes 0x0100
      # iff in little endian mode.
      self.COMM_ORDER = self._read_enum(fh, at(34))
      self.COMM_ORDER = self._read_enum(fh, at(34))

      self.TEMPLATE_NAME = self._read_string(fh, at(16))
      self.COMM_TYPE = self._read_enum(fh, at(32))

      self._WAVE_DESCRIPTOR_SIZE = self._read_long(fh, at(36))
      self._USER_TEXT_SIZE = self._read_long(fh, at(40))
      self._RES_DESC1_SIZE = self._read_long(fh, at(44))
      self._TRIGTIME_ARRAY_SIZE = self._read_long(fh, at(48))
      self._RIS_TIME_ARRAY_SIZE = self._read_long(fh, at(52))
      self._RES_ARRAY1_SIZE = self._read_long(fh, at(56))
      self._WAVE_ARRAY_1_SIZE = self._read_long(fh, at(60))

      # instrument info
      self.INSTRUMENT_NAME = self._read_string(fh, at(76))
      self.INSTRUMENT_NUMBER = self._read_long(fh, at(92))
      self.TRACE_LABEL = self._read_string(fh, at(96))

      self.WAVE_SOURCE = self._read_wave_source(fh, at(344))

      self.TRIG_TIME = self._read_timestamp(fh, at(296))

      self.RECORD_TYPE = self._read_record_type(fh, at(316))
      self.PROCESSING_DONE = self._read_processing_done(fh, at(318))

      self.TIMEBASE = self._read_timebase(fh, at(324))

      self.VERTICAL_GAIN = self._read_float(fh, at(156))
      self.VERTICAL_OFFSET = self._read_float(fh, at(160))

      self.VERTUNIT = self._read_string(fh, at(196))

      self.FIXED_VERT_GAIN = self._read_fixed_vert_gain(fh, at(332))

      self.HORIZ_INTERVAL = self._read_float(fh, at(176))
      self.HORIZ_OFFSET = self._read_double(fh, at(180))

      self.HORUNIT = self._read_string(fh, at(244))

      self.HORIZ_UNCERTAINTY = self._read_float(fh, at(292))

      self.VERT_COUPLING = self._read_vert_coupling(fh, at(326))

      self.PIXEL_OFFSET = self._read_double(fh, at(188))

      self.PNTS_PER_SCREEN = self._read_long(fh, at(120))
      self.FIRST_VALID_PNT = self._read_long(fh, at(124))
      self.LAST_VALID_PNT = self._read_long(fh, at(128))
      self.FIRST_POINT = self._read_long(fh, at(132))
      self.SPARSING_FACTOR = self._read_long(fh, at(136))
      self.SEGMENT_INDEX = self._read_long(fh, at(140))
      self.SUBARRAY_COUNT = self._read_long(fh, at(144))
      self.SWEEPS_PER_ACQ = self._read_long(fh, at(148))
      self.POINTS_PER_PAIR = self._read_word(fh, at(152))
      self.PAIR_OFFSET = self._read_word(fh, at(154))
      self.NOM_SUBARRAY_COUNT = self._read_word(fh, at(174))
      self.ACQ_DURATION = self._read_float(fh, at(312))
      self.RIS_SWEEPS = self._read_word(fh, at(322))
      self.PROBE_ATT = self._read_float(fh, at(328))

      self.MAX_VALUE = self._read_float(fh, at(164))
      self.MIN_VALUE = self._read_float(fh, at(168))
      self.NOMINAL_BITS = self._read_word(fh, at(172))
      self.BANDWIDTH_LIMIT = self._read_bandwidth_limit(fh, at(334))
      self.VERTICAL_VERNIER = self._read_float(fh, at(336))
      self.ACQ_VERT_OFFSET = self._read_float(fh, at(340))
      if self._USER_TEXT_SIZE > 0:
        self.USER_TEXT = self._read_string(fh, at(self._WAVE_DESCRIPTOR_SIZE),length=self._USER_TEXT_SIZE)
      else:
        self.USER_TEXT = ""

      self._payload_offset = at(self._WAVE_DESCRIPTOR_SIZE +
                          self._USER_TEXT_SIZE +
                          self._TRIGTIME_ARRAY_SIZE)

  @property
  def sampling_frequency(self):
    return 1/self.HORIZ_INTERVAL

  @property
  def LOFIRST(self):
    return not self.HIFIRST

  @property
  def HIFIRST(self):
    return self.COMM_ORDER == 0

  @property
  def WAVE_ARRAY_1(self):
    if self._WAVE_ARRAY_1 is None:
      self._WAVE_ARRAY_1 = self.read_wave_array(self._count)
    return self._WAVE_ARRAY_1

  @property
  def WAVE_ARRAY_RAW(self):
    if self._WAVE_ARRAY_RAW is None:
      self._WAVE_ARRAY_RAW = self.read_raw_data(self._count)
    return self._WAVE_ARRAY_RAW

  @property
  def WAVE_ARRAY_1_time(self):
    """
    A calculated array of when each sample in wave_form_1 was measured,
    based on HORIZ_OFFSET and HORIZ_INTERVAL.
    """
    tvec = np.arange(0, self._WAVE_ARRAY_1.size)
    return tvec * self.HORIZ_INTERVAL + self.HORIZ_OFFSET

  @property
  def metadata(self):
    """
    Returns a dictionary of metadata information.
    """
    metadict = dict()
    for name, value in vars(self).items():
      if not name.startswith('_') and name.isupper():
        metadict[name] = getattr(self, name)

    return metadict

  @property
  def mat(self):
    x = np.reshape(self.WAVE_ARRAY_1_time, (-1, 1))
    y = np.reshape(self.WAVE_ARRAY_1, (-1, 1))

    return np.column_stack((x,y))

  @property
  def comments(self):
    keyvaluepairs=list()
    for name, value in self.metadata.items():
      keyvaluepairs.append('%s=%s'%(name, value))
    return keyvaluepairs

  def savecsv(self, csvfname):
    """
    Saves the binary waveform as CSV, with metadata as headers.

    The header line will contain the string

      "LECROY BINARY WAVEFORM EXPORT"

    All headers will be prepended with '#'
    """
    mat = self.mat
    metadata = self.metadata
    jmeta = dict()
    for name, value in metadata.items():
      jmeta[name] = str(value)

    jmeta['EXPORTER'] = 'LECROY.PY'
    jmeta['AUTHOR'] = '@freespace'

    import json
    header = json.dumps(jmeta, sort_keys=True, indent=1)

    np.savetxt(csvfname, mat, delimiter=',', header=header)

  def _make_fmt(self, fmt):
    if self.HIFIRST:
      return '>' + fmt
    else:
      return '<' + fmt

  def _read(self, fh, addr, nbytes, fmt):
    fh.seek(addr)
    s = fh.read(nbytes)
    fmt = self._make_fmt(fmt)
    return np.fromstring(s, fmt)[0]

  def _read_byte(self, fh, addr):
    return self._read(fh, addr, 1, 'u1')

  def _read_word(self, fh, addr):
    return self._read(fh, addr, 2, 'i2')

  def _read_enum(self, fh, addr):
    return self._read(fh, addr, 2, 'u2')

  def _read_long(self, fh, addr):
    return self._read(fh, addr, 4, 'i4')

  def _read_float(self, fh, addr):
    return self._read(fh, addr, 4, 'f4')

  def _read_double(self, fh, addr):
    return self._read(fh, addr, 8, 'f8')

  def _read_string(self, fh, addr, length=16):
    result = self._read(fh, addr, length, 'S%d'%length)
    if sys.version_info > (3, 0):
      # Python 3 case
      result = result.decode('ascii')
    return result

  def _read_timestamp(self, fh, addr):
    second  = self._read_double(fh, addr)
    addr += 8 # double is 64 bits = 8 bytes

    minute  = self._read_byte(fh, addr)
    addr   += 1

    hour    = self._read_byte(fh, addr)
    addr   += 1

    day     = self._read_byte(fh, addr)
    addr   += 1

    month   = self._read_byte(fh, addr)
    addr   += 1

    year    = self._read_word(fh, addr)
    addr   += 2

    from datetime import datetime
    s = int(second)
    us = int((second - s) * 1000000)
    return datetime(year, month, day, hour, minute, s, us)

  def _read_vert_coupling(self, fh, addr):
    v = self._read_enum(fh, addr)
    coupling_desc = ['DC_50_Ohms',
                     'ground',
                     'DC_1MOhm',
                     'ground',
                     'AC,_1MOhm']
    return coupling_desc[v]

  def _read_processing_done(self, fh, addr):
    v = self._read_enum(fh, addr)
    processsing_desc = ['no_processing',
                        'fir_filter',
                        'interpolated',
                        'sparsed',
                        'autoscaled',
                        'no_result',
                        'rolling',
                        'cumulative']
    return processsing_desc[v]

  def _read_record_type(self, fh, addr):
    v = self._read_enum(fh, addr)
    record_types = ['single_sweep',
                    'interleaved',
                    'histogram',
                    'graph',
                    'filter_coefficient',
                    'complex',
                    'extrema',
                    'sequence_obsolete',
                    'centered_RIS',
                    'peak_detect']
    return record_types[v]

  def _read_timebase(self, fh, addr):
    v = self._read_enum(fh, addr)
    timebase = ['1_ps/div', '2_ps/div', '5_ps/div', '10_ps/div', '20_ps/div', '50_ps/div', '100_ps/div', '200_ps/div',
     '500_ps/div', '1_ns/div', '2_ns/div', '5_ns/div', '10_ns/div', '20_ns/div', '50_ns/div', '100_ns/div',
     '200_ns/div', '500_ns/div', '1_us/div', '2_us/div', '5_us/div', '10_us/div', '20_us/div', '50_us/div',
     '100_us/div', '200_us/div', '500_us/div', '1_ms/div', '2_ms/div', '5_ms/div', '10_ms/div', '20_ms/div',
     '50_ms/div', '100_ms/div', '200_ms/div', '500_ms/div', '1_s/div', '2_s/div', '5_s/div', '10_s/div', '20_s/div',
     '50_s/div', '100_s/div', '200_s/div', '500_s/div', '1_ks/div', '2_ks/div', '5_ks/div']
    if v == 1000:
      result = 'EXTERNAL'
    else:
      result = timebase[v]
    return result

  def _read_fixed_vert_gain(self, fh, addr):
    v = self._read_enum(fh, addr)
    fixed_vert_gain = ['1_uV/div','2_uV/div','5_uV/div','10_uV/div','20_uV/div','50_uV/div','100_uV/div',
                '200_uV/div','500_uV/div','1_mV/div','2_mV/div','5_mV/div','10_mV/div','20_mV/div',
                '50_mV/div','100_mV/div','200_mV/div','500_mV/div','1_V/div','2_V/div','5_V/div',
                '10_V/div','20_V/div','50_V/div','100_V/div','200_V/div','500_V/div','1_kV/div']
    return fixed_vert_gain[v]

  def _read_bandwidth_limit(self, fh, addr):
    v = self._read_enum(fh, addr)
    bandwith_limit = ['off', 'on']
    return bandwith_limit[v]

  def _read_wave_source(self, fh, addr):
    v = self._read_enum(fh, addr)
    wave_source = { 0 : 'CHANNEL_1', 1 : 'CHANNEL_2', 2: 'CHANNEL_3', 3: 'CHANNEL_4', 9 : 'UNKNOWN'}
    return wave_source.get(v, "")

  def read_raw_data(self, max_samples_count=-1):
    nbytes = self._WAVE_ARRAY_1_SIZE
    nsamples = nbytes
    max_bytes = max_samples_count
    if self.COMM_TYPE == 0:
      fmt = self._make_fmt('i1')
    else:
      fmt = self._make_fmt('i2')
      # if each sample is a 2 bytes, then we have
      # half as many samples as there are bytes in the wave
      # array
      nsamples //=2
      max_bytes *= 2
    if max_samples_count > 0:
      nsamples = min(nsamples, max_samples_count)
      nbytes = min(nbytes, max_bytes)

    addr = self._payload_offset
    dt = np.dtype((fmt, nsamples))
    s = None
    with _open(self._inputfilename, self._file_content) as fh:
      fh.seek(addr)
      s = fh.read(nbytes)
    data = np.fromstring(s, dtype=dt)[0]
    return data

  def read_wave_array(self, count=-1):
    # as per documentation, the actual value is gain * data - offset
    return self.VERTICAL_GAIN * self.WAVE_ARRAY_RAW - self.VERTICAL_OFFSET

def parse_commandline_arguments():
  import argparse
  parser = argparse.ArgumentParser(description='Reads binary Lecroy DSO traces and converts them to CSV')
  parser.add_argument('-csv',
                      action='store_true',
                      help='Converts inputs to csv. Outputs to the same filename with .csv appended')

  parser.add_argument('-trigtime',
                      action='store_true',
                      help='Print the trigtime of inputs')

  parser.add_argument('-metadata',
                      action='store_true',
                      help='Print the metadata (header)')

  parser.add_argument('-samples',
                      default=-1,
                      type=int,
                      help='number of samples to read')

  parser.add_argument('traces',
                      nargs='+',
                      help='Lecroy binary trc files')

  cmdargs = vars(parser.parse_args())
  return cmdargs

def main(**cmdargs):
  tracefiles = cmdargs['traces']
  print_trigtime = cmdargs['trigtime']
  print_metadata = cmdargs['metadata']
  convert_csv = cmdargs['csv']
  samples_count = cmdargs['samples']

  for tf in tracefiles:
    bwf = LecroyBinaryWaveform(tf, count=samples_count)

    if convert_csv:
      bwf.savecsv(tf+'.csv')

    if print_trigtime:
      print(bwf.TRIG_TIME)

    if print_metadata:
      for name, value in sorted(bwf.metadata.items()):
        print(name, value)

if __name__ == '__main__':
  import sys
  cmdargs = parse_commandline_arguments()
  sys.exit(main(**cmdargs))

